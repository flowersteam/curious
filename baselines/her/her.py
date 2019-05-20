import numpy as np
from baselines import logger


def make_sample_her_transitions(goal_replay, her_replay_k, reward_fun, task_replay='', tasks_ag_id=None, tasks_g_id=None):
    """Creates a sample function that can be used for HER experience replay.

    Args:
          goal_replay (in ['her', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used.
        her_replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if goal_replay == 'her':
        future_p = 1 - (1. / (1 + her_replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, task_to_replay=None, cp_proba=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        ag_id = []
        for t in range(len(tasks_g_id)):
            ag_id.extend(tasks_ag_id[t][:len(tasks_g_id[t])])
        transitions['g'][her_indexes] = future_ag[:, ag_id]

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['task_descr'] = None
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions



def make_sample_multi_task_her_transitions(goal_replay, her_replay_k, task_replay, reward_fun, tasks_ag_id=None, tasks_g_id=None):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        goal_replay (in ['her', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used.
        her_replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        task_replay: strategy to choose which task to replay (when the strucutre is modular)
        reward_fun (function): function to re-compute the reward with substituted goals
        tasks_ag_id: defines the indices of the different tasks inside an achieved goal vector.
        tasks_g_id: defines the indices of the different tasks inside an goal vector.

    """
    if goal_replay == 'her':
        future_p = 1 - (1. / (1 + her_replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    nb_tasks = len(tasks_ag_id)
    
    # if multiple buffers, the task to replay is not chosen by the sampling function, but by the strategy (DDPG object).
    if 'buffer' in task_replay or task_replay=='hand_designed':
        multiple_buffers = True
    else:
        multiple_buffers = False
    
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, task_to_replay=None, cp_proba=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0]
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        nb_replays = her_indexes.shape[0]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        changes = episode_batch['change'][episode_idxs[her_indexes], -1]


        for i in range(nb_replays):
            if task_replay != 'replay_current_task_transition':
                if multiple_buffers:
                    # task to replay is given by the id of the buffer from which we sample, to ensure activity in the replayed task
                    if task_to_replay is None:
                        replay_task = np.argwhere(transitions['task_descr'][her_indexes[i]]==1).squeeze()
                    else:
                        replay_task = task_to_replay

                elif task_replay == 'replay_random_task_transition':
                    replay_task = np.random.choice(range(nb_tasks))

                elif task_replay == 'replay_cp_task_transition':
                    replay_task = np.random.choice(range(nb_tasks), p=cp_proba)

                # replace task by replay_task, and replace goal by outcome achieved in the future, if goal_replay=='her'
                task_ag_id = tasks_ag_id[replay_task]
                task_g_id = tasks_g_id[replay_task]
                if len(task_ag_id) != len(task_g_id):
                    task_ag_id = task_ag_id[:len(task_g_id)]

                # clear previous goal and task_descr
                transitions['g'][her_indexes[i]] = 0
                transitions['task_descr'][her_indexes[i]] = 0
                # if her, replay goal and task
                transitions['g'][her_indexes[i]][task_g_id] = future_ag[i, task_ag_id]
                transitions['task_descr'][her_indexes[i]][replay_task] = 1

            # else no task replay
            else:
                task = np.argwhere(transitions['task_descr'][her_indexes[i]]==1).squeeze()
                task_ag_id = tasks_ag_id[task]
                task_g_id = tasks_g_id[task]
                if len(task_ag_id) != len(task_g_id):
                    task_ag_id = task_ag_id[:len(task_g_id)]
                transitions['g'][her_indexes[i], task_g_id] = future_ag[i, task_ag_id]


        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g', 'task_descr']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
        tasks = np.argwhere(transitions['task_descr']==1)[:,1]
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions