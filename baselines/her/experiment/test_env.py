import gym
import gym_flowers
import numpy as np
import os
os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'

# env = gym.make('ModularArm012-v0')
env = gym.make('MultiTaskFetchArm4-v5')
obs = env.reset()
goal = np.array([-1,-1,-1])
task = 2
env.unwrapped.reset_task_goal(goal, task)
env.render()
task_id = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14],[15,16,17],[18,19,20],[21,22,23], [24,25,26], [27,28,29],[30,31,32],[33,34,35]]
n_tasks = env.unwrapped.nb_tasks
for i in range(100000):
    act = np.array([1,0,0,0.1])
    obs = env.step(act)
    # print(obs[0]['observation'][9:15])
    for i in range(n_tasks):
        ag = obs[0]['achieved_goal']
        g = np.zeros([3 * n_tasks])
        g[task_id[i]] = obs[0]['desired_goal'][task_id[task]]
        task_descr = np.zeros([n_tasks])
        task_descr[i] = 1
        r = env.unwrapped.compute_reward(ag, g, task_descr=task_descr, info={})
        print('Task ', i, 'reward', r)
        # print(ag)
        # print(g)
    env.render()

