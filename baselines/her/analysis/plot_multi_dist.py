import json
import os
import random
import gym_flowers
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.signal import savgol_filter


os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'
import matplotlib
font = {'family' : 'normal',
        'size'   : 75}
matplotlib.rc('font', **font)


n_dists = [0,4,7]
# main_curve = 'avg' # 'median'
# error_type = 'std' # 'sem' #80
path_to_results = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/compare_dist' + '/'

strat = 'a'
gep_colors = [[0.3010,0.745,0.933],[0,0.447,0.7410],[222/255,21/255,21/255],[0.635,0.078,0.184],[0.466,0.674,0.188]]
matlab_colors = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.494,0.1844,0.556],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
matlab_colors2 = [[0.494,0.1844,0.556],[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
colors = matlab_colors


max_algo_n_eps = 0
algos_success = []
algo_all_results = []
fig1 = plt.figure(1, figsize=(40, 20), frameon=False)
fig2 = plt.figure(2, figsize=(40, 20), frameon=False)
plts = []

for i_d, n_dist in enumerate(n_dists):
    path_to_folder = path_to_results + str(n_dist) + '_' + strat

    competences = list()
    successes = list()

    for folder in os.listdir(path_to_folder):
        path_to_trial = path_to_folder + '/' + folder
        # extract params from json
        with open(path_to_trial + '/params.json') as json_file:
            params = json.load(json_file)
        try:
            structure = params['structure']
            task_selection = params['task_selection']
            goal_selection = params['goal_selection']
        except:
            structure = 'flat'
            task_selection = None
            goal_selection = 'random'

        n_cycles = params['n_cycles']
        rollout_batch_size = params['rollout_batch_size']
        n_cpu = params['num_cpu']


        # extract results
        data = pd.read_csv(path_to_trial + '/progress.csv')
        # extract episodes
        episodes = data['train/episode']
        n_epochs = len(episodes)
        n_eps = n_cpu * rollout_batch_size * n_cycles
        episodes = np.arange(n_eps, n_epochs * n_eps + 1, n_eps)
        episodes = episodes / 1000

        # extract test success rate
        test_success = data['test/success_rate'] * (4 + n_dist) / 4
        algos_success.append(test_success)

        # extract competence and cp
        n_points = test_success.shape[0]
        cp = np.zeros([n_points, 4])
        c = np.zeros([n_points, 4])
        if structure != 'flat':
            for i in range(4):
                c[:, i] = data['train/C_task' + str(i)]
                cp[:, i] = data['train/CP_task' + str(i)]
        competences.append(c[:208])
        successes.append(test_success[:208])
    comp = np.array(competences)
    c_plot = comp.mean(axis=0)
    success_plot = np.array(successes).mean(axis=0)
    plt.figure(1)
    for i in range(4):
        pl = plt.plot(episodes[:c_plot.shape[0]], c_plot[:, i], c=colors[i], linewidth=10)
        if i_d == 0:
            plts = plt.gca().get_lines()
    for i in range(4):
        plt.gca().get_lines()[-1-i].set_alpha(0.9 - 0.3 * i_d)
    plt.figure(2)
    plt.plot(episodes[:success_plot.shape[0]], success_plot, color=colors[i_d], linewidth=10)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.tick_params(width=4, direction='in', length=10, labelsize='small')
if strat == 'a':
    plt.title('Active task selection and replay')
else:
    plt.title('Random task selection and replay')
plt.ylabel('Agent subjective competence')
plt.xlabel('Episodes ($10^3$)')
plt.legend(plts, ['Task ' + str(i) for i in range(1,5)], frameon=False)
plt.savefig(path_to_results + strat + '_c.png', bbox_inches='tight')


fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)
ax.tick_params(width=4, direction='in', length=10, labelsize='small')
if strat == 'a':
    plt.title('Active task selection and replay')
else:
    plt.title('Random task selection and replay')
plt.xlabel('Episodes ($10^3$)')
plt.ylabel('Success rate\nover achievable tasks')
plt.legend([str(n_dist) + ' distractors' for n_dist in n_dists], frameon=False)
plt.savefig(path_to_results + strat + '_success.png', bbox_inches='tight')


