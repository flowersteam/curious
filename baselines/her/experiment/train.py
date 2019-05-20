import os
import sys
import datetime
import time
import argparse
import json
import numpy as np
from subprocess import CalledProcessError
from mpi4py import MPI

os.environ['LD_LIBRARY_PATH']=os.environ['HOME']+'/.mujoco/mjpro150/bin:'
sys.path.append('../../../')

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork, find_save_path, mpi_average
import psutil

ENV = 'MultiTaskFetchArm-v6'
NUM_CPU = 1

PERTURBATION_STUDY = False # Only in ENV = 'MultiTaskFetchArm4-v3'

STRUCTURE =  'curious'
# 'curious'
# 'flat'
# 'task_experts'  , must use TASK_REPLAY='replay_current_task_buffer'
TASK_SELECTION = 'active_competence_progress'
# 'random'
# 'active_competence_progress'
GOAL_SELECTION = 'random'
# 'random'
GOAL_REPLAY = 'her'
# 'her'
# 'none'
TASK_REPLAY = 'replay_task_cp_buffer'
# 'replay_current_task_transition'  # do not substitute the task
# 'replay_random_task_transition'   # substitute by a random task
# 'replay_cp_task_transition'       # substitute by a task depending on learning progress

# 'replay_task_random_buffer'       # pick a random task a randomly selected buffer (One buffer per task, transitions added if corresponding outcome moved)
# 'replay_task_cp_buffer'           # use a transition in the buffer selected using learning progress
# 'replay_current_task_buffer'      # use a transition in the buffer corresponding to the current task
t0 = time.time()


def train(policy, rollout_worker, evaluator, n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, structure, task_selection, params, perturbation_study, **kwargs):

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
        best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
        periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')
        logger.info("Training...")
    else:
        latest_policy_path = None
        best_policy_path = None
        periodic_policy_path = None
    best_success_rate = -1
    nb_tasks = params['nb_tasks']

    if structure == 'task_experts':
        p = 1 / nb_tasks * np.ones([nb_tasks])
        epoch = -1
        i_policy = -1
        evaluator.clear_history()
        evaluator.clear_competence_queue()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        best_success_rate = logs(rollout_worker[i_policy], evaluator, epoch, best_success_rate, best_policy_path, periodic_policy_path, policy_save_interval,
                                 save_policies, latest_policy_path, policy[i_policy], rank, structure, i_policy=i_policy, task_experts_cp=p)

        for epoch in range(n_epochs):

            if task_selection == 'random':
                # pick next task expert to train randomly
                i_policy = epoch % nb_tasks
            elif task_selection == 'active_competence_progress':
                # pick next task expert to train according to competence progress
                if rank == 0:
                    cps = []
                    for i in range(nb_tasks):
                        cp_i = np.array([rollout_worker[i].get_CP()]).squeeze()
                        cps.append(cp_i[i])
                    CP = np.array(cps).copy()
                    epsilon = params['eps_task']
                    if CP.sum() == 0:
                        proba = (1 / nb_tasks) * np.ones([nb_tasks])
                    else:
                        proba = epsilon * (1 / nb_tasks) * np.ones([nb_tasks]) + \
                                (1 - epsilon) * CP / CP.sum()

                    if p.sum() > 1:
                        p[np.argmax(p)] -= p.sum() - 1
                    elif p.sum() < 1:
                        p[-1] = 1 - p[:-1].sum()
                    i_policy = np.random.choice(range(nb_tasks), p=p)
                else:
                    i_policy = 0
                i_policy = MPI.COMM_WORLD.bcast(i_policy, root=0)

            # train
            rollout_worker[i_policy].clear_history()
            for _ in range(n_cycles):
                episode, cp, n_ep = rollout_worker[i_policy].generate_rollouts()
                policy[i_policy].store_episode(episode, cp, n_ep)
                for _ in range(n_batches):
                    policy[i_policy].train()
                policy[i_policy].update_target_net()

            # test
            evaluator.clear_history()
            for _ in range(n_test_rollouts):
                evaluator.generate_rollouts()

            best_success_rate = logs(rollout_worker[i_policy], evaluator, epoch, best_success_rate, best_policy_path, periodic_policy_path, policy_save_interval,
                 save_policies, latest_policy_path, policy[i_policy], rank, structure, i_policy=i_policy, task_experts_cp=p)

    else:
        # all tasks and goals learned by the same network (extended uvfa)
        epoch = -1
        i_policy = -1

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        best_success_rate = logs(rollout_worker, evaluator, epoch, best_success_rate, best_policy_path, periodic_policy_path, policy_save_interval,
                                 save_policies, latest_policy_path, policy, rank, structure)

        for epoch in range(n_epochs):
            logger.info('Starting new epoch ', epoch, 'at time', time.time()-t0)
            t_ep = time.time()
            # train
            rollout_worker.clear_history()

            if perturbation_study and epoch == 250:
                # In perturbation study, some observations starts to be biased at some point (break simulation)
                for i in range(2):
                    rollout_worker.envs[i].unwrapped.bias = True
                    evaluator.envs[i].unwrapped.bias = True

            for cyc in range(n_cycles):
                t_rol = time.time()
                episode, cp, n_ep = rollout_worker.generate_rollouts()
                policy.store_episode(episode, cp, n_ep)
                t_tr = time.time()
                for j in range(n_batches):
                    policy.train()
                policy.update_target_net()

            # test
            t_ev = time.time()
            evaluator.clear_history()
            for _ in range(n_test_rollouts):
                evaluator.generate_rollouts()
            logger.info('Epoch', epoch, 'over in ', time.time() - t_ep, 's.')


            best_success_rate = logs(rollout_worker, evaluator, epoch, best_success_rate, best_policy_path, periodic_policy_path,  policy_save_interval,
            save_policies, latest_policy_path, policy, rank, structure)



def logs(rollout_worker, evaluator, epoch, best_success_rate, best_policy_path, periodic_policy_path, policy_save_interval,
         save_policies, latest_policy_path, policy, rank, structure, i_policy=None, task_experts_cp=None):

    # record logs
    logger.record_tabular('epoch', epoch)
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, "%.3g" % mpi_average(val))
    for key, val in rollout_worker.logs('train'):
        logger.record_tabular(key, "%.3g" % mpi_average(val))
    for key, val in policy.logs():
        logger.record_tabular(key, "%.3g" % mpi_average(val))

    if rank == 0:
        if i_policy is not None:
            logger.record_tabular('IND_TASK_rollout', i_policy )
        for key, val in rollout_worker.additional_logs('train'):
            logger.record_tabular(key, val)
        for key, val in evaluator.additional_logs('test'):
            logger.record_tabular(key, val)
        # if task_experts_cp is not None:
        #     logger.record_tabular('multiple_expert_cp', task_experts_cp)
        logger.record_tabular('Time', time.time() - t0)
        logger.dump_tabular()
        rollout_worker.save_goal_task_history(logger.get_dir())

    # save the policy if it's better than the previous ones
    success_rate = mpi_average(evaluator.current_success_rate())
    if rank == 0 and success_rate >= best_success_rate and save_policies:
        best_success_rate = success_rate
        logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
        evaluator.save_policy(best_policy_path)
    if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
        policy_path = periodic_policy_path.format(epoch)
        logger.info('Saving periodic policy to {} ...'.format(policy_path))
        evaluator.save_policy(policy_path)
        evaluator.save_policy(latest_policy_path)

    # make sure that different threads have different seeds
    local_uniform = np.random.uniform(size=(1,))
    root_uniform = local_uniform.copy()
    MPI.COMM_WORLD.Bcast(root_uniform, root=0)
    if rank != 0:
        assert local_uniform[0] != root_uniform[0]

    return best_success_rate


def launch(env, trial_id, n_epochs, num_cpu, seed, policy_save_interval, clip_return, normalize_obs,
           structure, task_selection, goal_selection, goal_replay, task_replay, perturb, save_policies=True):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    # Configure logging
    if rank == 0:
        save_dir = find_save_path('./save/' + env + "/", trial_id)
        logger.configure(dir=save_dir)
    else:
        save_dir = None

    # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params, add main function arguments and log all parameters
    if structure == 'curious' or structure == 'task_experts':
        params = config.MULTI_TASK_PARAMS
    else:
        params = config.DEFAULT_PARAMS

    time = str(datetime.datetime.now())
    params['time'] = time
    params['env_name'] = env
    params['task_selection'] = task_selection
    params['goal_selection'] = goal_selection
    params['task_replay'] = task_replay
    params['goal_replay'] = goal_replay
    params['structure'] = structure
    params['normalize_obs'] = normalize_obs
    params['num_cpu'] = num_cpu
    params['clip_return'] = clip_return
    params['trial_id'] = trial_id
    params['seed'] = seed
    if rank == 0:
        with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
            json.dump(params, f)
    params = config.prepare_params(params)
    params['ddpg_params']['normalize_obs'] = normalize_obs
    if rank == 0:
        config.log_params(params, logger=logger)

    if num_cpu != 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Colas et al. (2018, https://arxiv.org/abs/1810.06284) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this.')
        logger.warn('****************')
        logger.warn()

    dims = config.configure_dims(params)

    buffers = config.configure_buffer(dims=dims, params=params)

    # creates several policies with shared buffers in the task-experts structure, otherwise use just one policy
    if structure == 'task_experts':
        policy = [config.configure_ddpg(dims=dims, params=params, buffers=buffers, clip_return=clip_return, t_id=i) for i in range(params['nb_tasks'])]
    else:
        policy = config.configure_ddpg(dims=dims, params=params, buffers=buffers, clip_return=clip_return)


    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'structure': structure,
        'task_selection': task_selection,
        'goal_selection': goal_selection,
        'queue_length': params['queue_length'],
        'eval': False,
        'eps_task': params['eps_task']
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'structure' : structure,
        'task_selection': task_selection,
        'goal_selection' : goal_selection,
        'queue_length': params['queue_length'],
        'eval': True,
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    if structure == 'task_experts':
        # create one rollout worker per policy/task
        rollout_worker = [RolloutWorker(params['make_env'], policy[i], dims, logger, unique_task=i, **rollout_params) for i in range(params['nb_tasks'])]
        for i in range(params['nb_tasks']):
            rollout_worker[i].seed(rank_seed + i)
    else:
        rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, **rollout_params)
        rollout_worker.seed(rank_seed)

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(rank_seed + 100)

    train(logdir=save_dir, policy=policy, rollout_worker=rollout_worker, evaluator=evaluator, n_epochs=n_epochs,
          n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'], perturbation_study=perturb,
          policy_save_interval=policy_save_interval, save_policies=save_policies, structure=structure, task_selection=task_selection, params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default=ENV, help='the name of the OpenAI Gym environment that you want to train on')
    parser.add_argument('--trial_id', type=int, default='0', help='trial identifier, name of the saving folder')
    parser.add_argument('--n_epochs', type=int, default=316, help='the number of training epochs to run')
    parser.add_argument('--num_cpu', type=int, default=NUM_CPU, help='the number of CPU cores to use (using MPI)')
    parser.add_argument('--seed', type=int, default=np.random.randint(int(1e6)), help='the random seed used to seed both the environment and the training code')
    parser.add_argument('--policy_save_interval', type=int, default=50, help='the interval with which policy are pickled.')
    parser.add_argument('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
    parser.add_argument('--normalize_obs', type=bool, default=False, help='whether observations are normalized or not')
    parser.add_argument('--structure', type=str, default=STRUCTURE, help='choose structure: curious, flat, multiple experts')
    parser.add_argument('--task_selection', type=str, default=TASK_SELECTION, help='choose task selection method: random, active with competence progress')
    parser.add_argument('--goal_selection', type=str, default=GOAL_SELECTION, help='choose the goal selection method: only random is implemented')
    parser.add_argument('--task_replay', type=str, default=TASK_REPLAY, help='strategy for task replay (see options at the top)')
    parser.add_argument('--goal_replay', type=str, default=GOAL_REPLAY, help='"her" uses HER, "none" disables HER.')
    parser.add_argument('--perturb', type=bool, default=PERTURBATION_STUDY, help='"her" uses HER, "none" disables HER.')
    kwargs = vars(parser.parse_args())
    launch(**kwargs)