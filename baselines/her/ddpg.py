from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea
from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.normalizer import Normalizer
from baselines.common.mpi_adam import MpiAdam
from mpi4py import MPI
import pickle
import time

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG(object):
    @store_args
    def __init__(self, input_dims, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 normalize_obs, sample_transitions, gamma, buffers=None, reuse=False, tasks_ag_id=None, tasks_g_id=None, task_replay='', t_id=None,
                 eps_task=None, **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused,
            buffers (list): buffers to be used to store new transition (usually one per task + 1
            task_ag_id (list): indices to find achieved goals for each task in the achieved goal vector
            task_g_id (list): indices to find agoals for each task in the goal vector
            task_replay (str): defines the task replay strategy (see train.py for info)
            t_id (int): index of the task corresponding to this policy when using a task-experts structure
            eps_task (float): epsilon parameter for the epsilon greedy strategy (task choice)
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)
        self.normalize_obs = normalize_obs

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimag = self.input_dims['ag']
        self.dimu = self.input_dims['u']
        if self.structure == 'curious' or self.structure == 'task_experts':
            self.dimtd = self.input_dims['task_descr']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None, 1)
        self.stage_shapes = stage_shapes

        if t_id is not None:
            self.scope += str(t_id)
        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # addition for multi-task structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            self.tasks_g_id = tasks_g_id
            self.tasks_ag_id = tasks_ag_id
            self.nb_tasks = len(tasks_g_id)

        if buffers is not None:
            self.buffer = buffers
            if type(self.buffer) is list:
                if len(self.buffer) > 5:
                    # distractor buffers are equal
                    for i in range(6, len(self.buffer)):
                        self.buffer[i] = self.buffer[5]
        self.first = True


    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))


    def _preprocess_og(self, o, ag, g):
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimag)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, task_descr=None, noise_eps=0., random_eps=0., use_target_net=False, compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }
        # addition for multi-task structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            feed[policy.td_tf] = task_descr.reshape(-1, self.dimtd)

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, cp, n_ep, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """
        # decompose episode_batch in episodes
        batch_size = episode_batch['ag'].shape[0]
        # addition in the case of curious goals, compute count of achieved goal that moved in the n modules
        self.cp = cp
        self.n_episodes = n_ep
        # addition for multi-task structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            new_count_local = np.zeros([self.nb_tasks])
            new_count_total = np.zeros([self.nb_tasks])
            # add a new transition in a buffer only if the corresponding outcome has changed compare to the initial outcome
            for b in range(batch_size):
                active_tasks = []
                for j in range(self.nb_tasks):
                    if any(episode_batch['change'][b, -1, self.tasks_ag_id[j][:len(self.tasks_g_id[j])]]):
                        new_count_local[j] += 1
                        if self.nb_tasks < 5 or j < 5:
                            active_tasks.append(j)
                MPI.COMM_WORLD.Allreduce(new_count_local, new_count_total, op=MPI.SUM)
                ep = dict()
                for key in episode_batch.keys():
                    ep[key] = episode_batch[key][b].reshape([1, episode_batch[key].shape[1], episode_batch[key].shape[2]])

                if 'buffer' in self.task_replay or self.task_replay == 'hand_designed':
                    if len(active_tasks) == 0:
                        ind_buffer = [0]
                    else:
                        for task in active_tasks:
                            self.buffer[task+1].store_episode(ep)
                else:
                    self.buffer.store_episode(ep)

        elif self.structure == 'flat' or self.structure == 'task_experts':
            for b in range(batch_size):
                ep = dict()
                for key in episode_batch.keys():
                    ep[key] = episode_batch[key][b].reshape([1, episode_batch[key].shape[1], episode_batch[key].shape[2]])
                self.buffer.store_episode(ep)

        # update statistics for goal and observation normalizations
        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            if self.structure == 'curious' or self.structure == 'task_experts':
                transitions = self.sample_transitions(episode_batch, num_normalizing_transitions, task_to_replay=None)
            else:
                transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)
            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])
            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()


    def get_current_buffer_size(self):
        return sum([self.buffer[i].get_current_size() for i in range(self.nb_tasks)])


    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()


    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        return critic_loss, actor_loss, Q_grad, pi_grad


    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)


    def sample_batch(self):
        # addition for multi-task structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            if self.structure == 'curious':
                if 'buffer' in self.task_replay or self.task_replay == 'hand_designed':
                    buffers_sizes = np.array([self.buffer[i].current_size * self.T for i in range(self.nb_tasks + 1)])
                    self.proportions = np.zeros([self.nb_tasks + 1])
                    buffers_sizes = np.array([self.buffer[i].current_size * self.T for i in range(self.nb_tasks + 1)])
                    self.proportions = np.zeros([self.nb_tasks + 1])
                    if buffers_sizes[1:].sum() < self.T:
                        ind_valid_buffers = np.array([0])
                        n_valid = 1
                        self.proportions = buffers_sizes / buffers_sizes.sum() * self.batch_size
                    else:
                        ind_valid_buffers = np.argwhere(buffers_sizes[1:] > 0)
                        ind_valid_buffers = ind_valid_buffers.reshape([ind_valid_buffers.size])
                        n_valid = len(ind_valid_buffers)

                        # draw transition from random buffers (random tasks)
                        if self.task_replay == 'replay_task_random_buffer':
                            proba = 1 / ind_valid_buffers.size * np.ones([n_valid])
                        elif self.task_replay == 'replay_task_cp_buffer':
                            CP = self.cp[ind_valid_buffers]
                            if CP.sum() == 0:
                                proba = (1 / n_valid) * np.ones([n_valid])
                            else:
                                proba = self.eps_task * (1 / n_valid) * np.ones([n_valid]) + \
                                         (1 - self.eps_task) * CP / CP.sum()
                            proba[-1] = 1 - proba[:-1].sum()
                        self.proportions[ind_valid_buffers + 1] = proba * self.batch_size

                    self.proportions = self.proportions.astype(np.int)
                    remain = self.batch_size - self.proportions.sum()
                    for i in range(remain):
                        self.proportions[ind_valid_buffers[i % n_valid] + 1] += 1
                    self.proportions = self.proportions.astype(np.int)

                elif self.task_replay == 'replay_cp_task_transition':
                    CP = self.cp.copy()
                    if CP.sum() == 0:
                        proba = (1 / self.nb_tasks) * np.ones([self.nb_tasks])
                    else:
                        proba = self.eps_task * (1 / self.nb_tasks) * np.ones([self.nb_tasks]) + \
                                (1 - self.eps_task) * CP / CP.sum()
                    proba[-1] = 1 - proba[:-1].sum()
                    transitions = self.buffer.sample(self.batch_size,  task_to_replay=None, cp_proba=proba)
    
                else:
                    transitions = self.buffer.sample(self.batch_size, task_to_replay=None, cp_proba=None)


            elif self.structure == 'task_experts':
                if self.task_replay == 'replay_current_task_buffer':
                    buffers_sizes = np.array([self.buffer[i].current_size * self.T for i in range(self.nb_tasks + 1)])
                    ind_valid_buffers = np.argwhere(buffers_sizes > 0)
                    ind_valid_buffers = ind_valid_buffers.reshape([ind_valid_buffers.size])
                    n_valid = len(ind_valid_buffers)
                    self.proportions = np.zeros([self.nb_tasks + 1])
                    if buffers_sizes[self.t_id+1] > 0:
                        self.proportions[self.t_id+1] = 1
                    else:
                        self.proportions[ind_valid_buffers] = 1 / len(ind_valid_buffers)
                    self.proportions *= self.batch_size
                    self.proportions = self.proportions.astype(np.int)
                    remain = self.batch_size - self.proportions.sum()
                    for i in range(remain):
                        self.proportions[ind_valid_buffers[i % n_valid]] += 1
                    self.proportions = self.proportions.astype(np.int)
                else:
                    transitions = self.buffer.sample(self.batch_size, task_to_replay=None, cp_proba=None)

            if 'buffer' in self.task_replay or self.task_replay == 'hand_designed':
                assert self.proportions.sum() == self.batch_size

                # sample transitions from different buffers
                trans = []
                for i in range(self.nb_tasks + 1):
                    if self.proportions[i] > 0:
                        if self.structure == 'curious':
                            if i > 0:
                                task_to_replay = i-1
                            else:
                                task_to_replay = None
                        else:
                            task_to_replay = self.t_id
                        trans.append(self.buffer[i].sample(self.proportions[i], task_to_replay=task_to_replay))
                # concatenate transitions from different buffers and shuffle
                shuffle_inds = np.arange(self.batch_size)
                np.random.shuffle(shuffle_inds)
                transitions = dict()
                for key in trans[0].keys():
                    tmp = np.array([]).reshape([0, trans[0][key].shape[1]])
                    for ts in trans:
                        tmp = np.concatenate([tmp, ts[key]])
                    transitions[key] = tmp[shuffle_inds, :]

        elif self.structure == 'flat':
            transitions = self.buffer.sample(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)

        # #test addition !!
        # transitions['task_descr'] = np.zeros([256, 8])

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        for i in range(self.nb_tasks):
            self.buffer[i].clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info("g a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        # networks
        with tf.variable_scope('main') as vs:
            if reuse:
                vs.reuse_variables()
            self.main = self.create_actor_critic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target') as vs:
            if reuse:
                vs.reuse_variables()
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = self.create_actor_critic(
                target_batch_tf, net_type='target', **self.__dict__)
            vs.reuse_variables()
        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0. if self.clip_pos_returns else np.inf)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def save_weights(self, path):
        to_save = []
        scopes_var = ['main/Q', 'main/pi', 'target/Q', 'target/pi']
        scopes_global_var = ['o_stats', 'g_stats']
        for s in scopes_var:
            tmp = []
            for v in self._vars(s):
                tmp.append(v.eval())
            to_save.append(tmp)
        for s in scopes_global_var:
            tmp = []
            for v in self._global_vars(s):
                tmp.append(v.eval())
            to_save.append(tmp)

        with open(path+'_weights.pkl', 'wb') as f:
            pickle.dump(to_save, f)

    def load_weights(self, path):
        with open(path + '_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        scopes_var = ['main/Q', 'main/pi', 'target/Q', 'target/pi']
        scopes_global_var = ['o_stats', 'g_stats']
        for i_s, s in enumerate(scopes_var):
            for i_v, v in enumerate(self._vars(s)):
                v.load(weights[i_s][i_v])
        for i_s, s in enumerate(scopes_global_var):
            for i_v, v in enumerate(self._global_vars(s)):
                v.load(weights[i_s + len(scopes_var)][i_v])

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        # state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
