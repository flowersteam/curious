import os
import pickle
import json
import argparse
import numpy as np

os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker



PATH = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/MultiTaskFetchArm4-v3/380' + '/'
POLICY_FILE = PATH + 'policy_300.pkl'
PARAMS_FILE = PATH + 'params.json'


def play(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Load params
    with open(PARAMS_FILE) as json_file:
        params = json.load(json_file)

    params['env_name'] = env_name

    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    structure = params['structure']
    task_selection = params['task_selection']
    goal_selection = params['goal_selection']

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'structure': structure,
        'task_selection': task_selection,
        'goal_selection': goal_selection,
        'queue_length': params['queue_length'],
        'eval': True,
        'render': render
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    for _ in range(n_test_rollouts):
        evaluator.generate_rollouts()

    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy_file', type=str, default=POLICY_FILE)
    parser.add_argument('--seed', type=int, default=int(np.random.randint(1e6)))
    parser.add_argument('--n_test_rollouts', type=int, default=30)
    parser.add_argument('--render', type=int, default=1)
    kwargs = vars(parser.parse_args())
    play(**kwargs)
