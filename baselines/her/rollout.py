from collections import deque
from mpi4py import MPI
import numpy as np
import pickle
from mujoco_py import MujocoException
from collections import deque
from baselines.her.util import convert_episode_to_batch_major, store_args
from baselines.her.queues import CompetenceQueue
from baselines import logger

from baselines.her.active_goal_sampling import SAGG_RIAC

class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1, exploit=False, use_target_net=False, compute_Q=False,
                 noise_eps=0, random_eps=0, history_len=100, render=False , structure='curious', task_selection='random', goal_selection='random',
                 queue_length=500, eval=False, unique_task=None, temperature=None, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rolloutscd
            structure (str): structure of the agent ('multi_experts', 'flat' or 'curious')
            goal_selection (str): 'random', 'active', strategy to sample next goal in given task
            task_selection (str): 'random', 'active_competence_progress'
            eval (boolean): whether the rollout worker is for evaluation or not
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.reward_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0 # episode counter
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['ag']), np.float32)  # achieved goals

        self.rank = MPI.COMM_WORLD.Get_rank() # rank of cpu
        self.nb_cpu = MPI.COMM_WORLD.Get_size()
        self.nb_goals_per_rollout = self.nb_cpu * self.rollout_batch_size
        self.nb_tasks = self.envs[0].unwrapped.nb_tasks
        self.C = np.zeros([self.nb_tasks])  # competence
        self.CP = np.zeros([self.nb_tasks]) # comptence progress or learning progress

        # addition for curious and task-experts structures
        if self.structure == 'curious' or self.structure == 'task_experts':

            self.tasks_ag_id = self.envs[0].unwrapped.tasks_ag_id  # indices of achieved goals for each task
            self.tasks_g_id = self.envs[0].unwrapped.tasks_g_id  # indices of goals for each task

            self.task_descr = np.empty((self.rollout_batch_size, self.nb_tasks), np.float32)  # place holder for task descriptors
            self.p = 1 / self.nb_tasks * np.ones([self.nb_tasks]) # selection probabilities for each task (random at first)

            # In task-experts configuration, each rollout worker is assigned to a particular task.
            if self.structure == 'task_experts' and not self.eval:
                self.p = np.zeros([self.nb_tasks])
                self.p[unique_task] = 1

            self.competence_computers = [CompetenceQueue(window=queue_length) for _ in range(self.nb_tasks)]
            self.task_history = deque()
            self.goal_history = deque()
            self.compute_progress = False
            self.split_histories = [deque() for i in range(self.nb_tasks)]

            if self.goal_selection == 'active':
                min_goal_spaces = [None] * self.nb_tasks
                max_goal_spaces = [None] * self.nb_tasks
                for i in range(self.nb_tasks):
                    min_goal_spaces[i] = self.envs[0].unwrapped._compute_goal(- np.ones([len(self.tasks_g_id[i])]), i, eval=False)[0][self.tasks_g_id[i]]
                    max_goal_spaces[i] = self.envs[0].unwrapped._compute_goal(np.ones([len(self.tasks_g_id[i])]), i, eval=False)[0][self.tasks_g_id[i]]
                self.goal_selectors = [SAGG_RIAC(min_goal_spaces[i], max_goal_spaces[i]) for i in range(self.nb_tasks)]



        # if flat structure, turn the environment to its flat structure
        elif self.structure == 'flat':
            for i in range(self.rollout_batch_size):
                self.envs[i].unwrapped.set_flat_env()

        self.stochastic_reset = False # if true, do not reset at every episode
        self.count = -1
        self.reset_all_rollouts()
        self.clear_history()


    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new task and a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if self.eval or not self.stochastic_reset or np.random.rand() < 0.3 or self.exploit:
            obs = self.envs[i].reset()
            obs = obs['observation']
        else:
            obs = self.envs[i].unwrapped.last_obs.copy()

        # here we first reset the environment to get the initial state, then we call the goal sampler to sample the next goal
        # finally, we set the goal and task to the environment.
        if self.structure == 'curious' or self.structure == 'task_experts':
            tasks_to_scatter = []
            goals_to_scatter = []
            active_goal = None
            if self.rank == 0:
                # sample tasks and goals for all cpus
                tasks = np.random.choice(range(self.nb_tasks), p=self.p, size=self.nb_cpu).tolist()
                if self.goal_selection == 'active' and not self.eval:
                    goals = []
                    for ta in tasks:
                        g = self.goal_selectors[ta].sample_goal()
                        goals.append(g)
                    active_goal = True
                else:
                    active_goal = False
                    goals = [np.random.uniform(-1, 1, len(self.tasks_g_id[tasks[i]])) for i in range(self.nb_cpu)]
                tasks_to_scatter.extend(tasks)
                goals_to_scatter.extend(goals)
                for cpu in range(self.nb_cpu):
                    good_ind = cpu * self.rollout_batch_size + i
                    self.tasks[good_ind] = tasks[cpu]
                    self.goals[good_ind] = self.envs[i].unwrapped._compute_goal(goals[cpu], tasks[cpu], eval=self.eval)[0][self.tasks_g_id[tasks[cpu]]]

            # redistribute all goals, tasks, params, random indicator to the cpus
            active_goal = MPI.COMM_WORLD.bcast(active_goal, root=0)
            task = MPI.COMM_WORLD.scatter(tasks_to_scatter, root=0)
            goal = MPI.COMM_WORLD.scatter(goals_to_scatter, root=0)
            self.count += 1

            obs = self.envs[i].unwrapped.reset_task_goal(goal=goal, task=task, directly=active_goal, eval=self.eval)

        elif self.structure == 'flat':
            goals = []
            if self.rank==0:
                goals = [np.random.uniform(-1, 1, self.dims['g']) for _ in range(self.nb_cpu)]
                for cpu in range(self.nb_cpu):
                    good_ind = cpu * self.rollout_batch_size + i
                    self.goals[good_ind] = self.envs[i].unwrapped._compute_goal(goals[cpu], 0)[0]

            goal = MPI.COMM_WORLD.scatter(goals, root=0)
            obs = self.envs[i].unwrapped.reset_task_goal(goal=goal)

        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']

        # addition for multi-task structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            self.task_descr[i] = obs['mask']


    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        self.goals = [[] for i in range(self.nb_goals_per_rollout)]
        # addition for multi-tasks structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            self.tasks = [[] for i in range(self.nb_goals_per_rollout)]

        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)


    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        # addition for multi-tasks structures
        # decides whether the next runs are made to compute progress (exploit = True, means no noise on actions).
        if (self.structure == 'curious' or self.structure == 'task_experts') and not self.eval:
            self.exploit = True if np.random.random() < 0.1 else False
            if self.exploit and self.structure == 'curious':
                    self.p = 1 / self.nb_tasks * np.ones([self.nb_tasks])
        elif self.eval:
            self.exploit = True
            self.p = 1 / self.nb_tasks * np.ones([self.nb_tasks])

        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['ag']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        # addition for multi-tasks structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            task_descrs = []
            changes = []  # True when the achieved goal (outcome) has changed compared to the initial achieved goal

        for t in range(self.T):

            # when evaluating task_experts, the policy corresponding to the demanded task must be selected
            if self.structure=='task_experts' and self.eval:
                act_output = np.zeros([self.rollout_batch_size, self.dims['u']])
                q_output = np.zeros([self.rollout_batch_size, 1])
                for i in range(self.rollout_batch_size):
                    tsk = np.argwhere(self.task_descr[i] == 1).squeeze()
                    act_output[i, :], q_output[i, 0] = self.policy[tsk].get_actions(
                        o[i].reshape([1, o[i].size]), ag[i].reshape([1, ag[i].size]), self.g[i].reshape([1, self.g[i].size]),
                        task_descr=self.task_descr[i].reshape([1, self.task_descr[i].size]),
                        compute_Q=self.compute_Q,
                        noise_eps=self.noise_eps if not self.exploit else 0.,
                        random_eps=self.random_eps if not self.exploit else 0.,
                        use_target_net=self.use_target_net)
                policy_output = [act_output, q_output]
            else:
                policy_output = self.policy.get_actions(
                    o, ag, self.g,
                    task_descr = self.task_descr if self.structure == 'curious' else None,
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['ag']))
            success = np.zeros(self.rollout_batch_size)
            r_competence = np.zeros(self.rollout_batch_size)

            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    if self.render:
                        self.envs[i].render()
                    curr_o_new, r_competence[i], _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    self.g[i] = curr_o_new['desired_goal'] # in case desired goal changes depending on observation
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]

                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new

            # addition for goal task selection
            if self.structure == 'curious' or self.structure == 'task_experts':
                task_descrs.append(self.task_descr.copy())
                changes.append(np.abs(achieved_goals[0] - ag) > 1e-3)


        obs.append(o.copy())
        achieved_goals.append(ag.copy())

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)

        # addition for multi-tasks structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            episode['task_descr'] = task_descrs
            episode['change'] = changes


        self.initial_o[:] = o
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        self.reward_history.append(r_competence)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size * self.nb_cpu

        # addition for multi-tasks structures
        if self.structure == 'curious' or self.structure == 'task_experts':
            # only update competence if no noise has been used
            if self.exploit:
                tasks_for_competence = [self.envs[i].unwrapped.task for i in range(self.rollout_batch_size)]
                goals_for_competence = [self.envs[i].unwrapped.goal[self.tasks_g_id[tasks_for_competence[i]]] for i in range(self.rollout_batch_size)]
                full_goals_for_competence = [self.envs[i].unwrapped.goal for i in range(self.rollout_batch_size)]
                ag_for_competence = [achieved_goals[-1][i] for i in range(self.rollout_batch_size)]

                succ_list = successful.tolist()
            else:
                tasks_for_competence = []
                goals_for_competence = []
                full_goals_for_competence = []
                ag_for_competence = []
                succ_list = []

            succ_list = MPI.COMM_WORLD.gather(succ_list, root=0)
            tasks_for_competence = MPI.COMM_WORLD.gather(tasks_for_competence, root=0)
            goals_for_competence = MPI.COMM_WORLD.gather(goals_for_competence, root=0)
            full_goals_for_competence = MPI.COMM_WORLD.gather(full_goals_for_competence, root=0)
            ag_for_competence = MPI.COMM_WORLD.gather(ag_for_competence, root=0)

            # update competence queues for each task in cpu rank 0
            # compute next selection probabilities
            if self.rank == 0:
                tasks_for_competence = sum(tasks_for_competence, [])
                goals_for_competence = sum(goals_for_competence, [])
                succ_list = sum(succ_list, [])

                task_succ_list = [[] for _ in range(self.nb_tasks)]
                task_cp_list = [[] for _ in range(self.nb_tasks)]
                task_goal_list = [[] for _ in range(self.nb_tasks)]
                # update competence queues
                for succ, task in zip(succ_list, tasks_for_competence):
                    task_succ_list[task].append(succ)

                for goal, task in zip(goals_for_competence, tasks_for_competence):
                    task_goal_list[task].append(goal)

                for task in range(self.nb_tasks):
                    self.competence_computers[task].update(task_succ_list[task]) # update competence and competence progress (learning progress)
                    if self.goal_selection == 'active' and not self.eval:

                        new_split, _ = self.goal_selectors[task].update(task_goal_list[task], task_succ_list[task])
                        if new_split:
                            regions = self.goal_selectors[task].get_regions
                            probas = self.goal_selectors[task].probas
                            self.split_histories[task].append([regions, probas])
                        else:
                            self.split_histories[task].append(None)

                self.C = np.array([self.get_C()]).squeeze() # get new updated competence measures

                # record all tasks
                self.task_history.extend(self.tasks.copy())
                self.goal_history.extend(self.goals.copy())

                # update task selection probabilities if active task selection
                if not self.eval:
                    if self.task_selection == 'active_competence_progress' and self.structure != 'task_experts':
                        # compute competence progress for each task
                        self.CP = np.array([self.get_CP()]).squeeze()
                        # softmax
                        # exp_cp = np.exp(self.temperature*self.CP)
                        # self.p = exp_cp / exp_cp.sum()

                        # epsilon proportional
                        epsilon = 0.4
                        if self.CP.sum() == 0:
                            self.p = (1 / self.nb_tasks) * np.ones([self.nb_tasks])
                        else:
                            self.p = epsilon * (1 / self.nb_tasks) * np.ones([self.nb_tasks]) + \
                                     (1 - epsilon) * self.CP / self.CP.sum()

                        if self.p.sum() > 1:
                            self.p[np.argmax(self.p)] -= self.p.sum() - 1
                        elif self.p.sum() < 1:
                            self.p[-1] = 1 - self.p[:-1].sum()


                    elif self.structure == 'task_experts':
                        self.p = np.zeros([self.nb_tasks])
                        self.p[self.unique_task] = 1


            # broadcast the selection probability to all cpus and the competence
            if not self.eval:
                self.p = MPI.COMM_WORLD.bcast(self.p, root=0)
                self.CP = MPI.COMM_WORLD.bcast(self.CP, root=0)

        return convert_episode_to_batch_major(episode), self.CP, self.n_episodes

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.reward_history.clear()
        self.Q_history.clear()

    def clear_competence_queue(self):
        for i in range(self.nb_tasks):
            self.competence_computers[i].clear_queue()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)
        try:
            self.policy.save_weights(path)
        except:
            pass
            # for i, p in enumerate(self.policy):
            #     p.save_weights(path + str(i))

    def save_goal_task_history(self, path):
        # if self.structure == 'curious' or self.structure == 'task_experts':
        #     dict_out = dict(tasks=np.array(self.task_history),
        #                     goals=np.array(self.goal_history),
        #                     splits=self.split_histories)
        #
        #     if self.unique_task is not None:
        #         with open(path + 'extra_saves_' + str(self.unique_task) + '.pk', 'wb') as f:
        #             pickle.dump(dict_out, f)
        #     else:
        #         with open(path + 'extra_saves.pk', 'wb') as f:
        #             pickle.dump(dict_out, f)
        pass

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('avg_reward', np.mean(self.reward_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def additional_logs(self, prefix='worker'):
        logs = []
        if self.structure == 'curious' or self.structure == 'task_experts':
            for i in range(self.nb_tasks):
                Cs = self.get_C()
                logs += [('C_task' + str(i), "%.3g" % Cs[i])]
                if not self.eval:
                    CPs = self.get_CP()
                    logs += [('CP_task' + str(i), "%.3g" % CPs[i])]
                    per = np.mean(np.array(self.task_history)[-100:] == i)
                    logs += [('%_task' + str(i), "%.3g" % per)]
                    logs += [('p_task' + str(i), "%.3g" % self.p[i])]


        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def get_CP(self):
        # addition for active goal task selection
        # extract measures of competence progress for all tasks
        return [cq.CP for cq in self.competence_computers]

    def get_C(self):
        # addition for active goal task selection
        # extract measures of competence for all tasks
        return [cq.C for cq in self.competence_computers]

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)


