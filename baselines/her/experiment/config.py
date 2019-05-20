import numpy as np
import gym
import gym_flowers

from baselines import logger
from baselines.her.ddpg import DDPG
from baselines.her.her import make_sample_her_transitions, make_sample_multi_task_her_transitions
from baselines.her.util import import_function
from baselines.her.replay_buffer import ReplayBuffer



DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 25,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 100,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 5,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'her_replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'her_sampling_func': 'baselines.her.her:make_sample_her_transitions',
    'queue_length': 200,
}

MULTI_TASK_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:MultiTaskActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 25,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 100, #40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 5,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'her_replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'her_sampling_func': 'baselines.her.her:make_sample_multi_task_her_transitions',
    'queue_length': 300, # length of queue for computation of competence
    'eps_task': 0.4  # epsilon greedy parameter for active task choice
}



CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    def make_env():
        return gym.make(env_name)
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    if kwargs['structure'] == 'flat':
        tmp_env.unwrapped.set_flat_env()
    kwargs['nb_tasks'] = tmp_env.unwrapped.nb_tasks
    kwargs['tasks_g_id'] = tmp_env.unwrapped.tasks_g_id
    kwargs['tasks_ag_id'] = tmp_env.unwrapped.tasks_ag_id
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    # if kwargs['num_cpu'] == 1:
    #     # make more test rollout when number of cpu is lower (to get a better estimate)
    #     kwargs['n_test_rollouts'] = 100
    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def configure_her(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    if params['structure'] == 'flat':
        env.unwrapped.set_flat_env()

    def reward_fun(ag_2, g, task_descr, info):  # vectorized
        return env.unwrapped.compute_reward(achieved_goal=ag_2, goal=g, task_descr=task_descr, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
        'tasks_ag_id': params['tasks_ag_id'],
        'tasks_g_id': params['tasks_g_id'],
        'goal_replay': params['goal_replay'],
        'her_replay_k': params['her_replay_k'],
        'task_replay': params['task_replay']
    }

    her_sampling_func = import_function(params['her_sampling_func'])
    sample_her_transitions = her_sampling_func(**her_params)

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}

def configure_buffer(dims, params):
    T = params['T']
    structure = params['structure']
    buffer_size = params['buffer_size']
    rollout_batch_size = params['rollout_batch_size']
    task_replay = params['task_replay']
    sample_her_transitions = configure_her(params)

    input_shapes = dims_to_shapes(dims)
    
    dimg = dims['g']
    dimag = dims['ag']
    if structure == 'curious' or structure == 'task_experts':
        dimtask_descr = dims['task_descr']
        
    # Configure the replay buffer.
    buffer_shapes = {key: (T if key != 'o' else T + 1, *input_shapes[key])
                     for key, val in input_shapes.items()}
    buffer_shapes['g'] = (buffer_shapes['g'][0], dimg)
    buffer_shapes['ag'] = (T + 1, dimag)
    buffer_size = (buffer_size // rollout_batch_size) * rollout_batch_size

    # addition for goal module selection
    buffer_shapes['task_descr'] = (buffer_shapes['g'][0], dimtask_descr)
    buffer_shapes['change'] = (buffer_shapes['g'][0], dimag)

    if 'buffer' in task_replay:
        # use several buffer, each corresponds to a task, the first corresponds to transition where no outcome moved.
        buffers = [ReplayBuffer(buffer_shapes, buffer_size, T, sample_her_transitions) for i in range(params['nb_tasks'] + 1)]
    else:
        buffers = ReplayBuffer(buffer_shapes, buffer_size, T, sample_her_transitions)

    return buffers


def configure_ddpg(dims, params, buffers, reuse=False, use_mpi=True, clip_return=True, t_id=None):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        'task_replay': params['task_replay'],
                        'structure': params['structure'],
                        'tasks_ag_id': params['tasks_ag_id'],
                        'tasks_g_id': params['tasks_g_id'],
                        'eps_task': params['eps_task']
                        })

    if t_id is not None:
        # give task id to rollout worker in the case of multiple task-experts
        ddpg_params.update({'t_id':t_id})

    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    policy = DDPG(reuse=reuse, **ddpg_params, buffers=buffers, use_mpi=use_mpi)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    info = env.unwrapped.info

    dims = {
        'o': env.observation_space.spaces['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': env.observation_space.spaces['desired_goal'].shape[0],
        'ag': env.observation_space.spaces['achieved_goal'].shape[0]
    }
    # addition in the case of curious structure
    dims['task_descr'] = params['nb_tasks']

    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
