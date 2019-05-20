import json
import os
import random
import gym_flowers
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon
from scipy.signal import savgol_filter


os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'
import matplotlib
font = {'family' : 'normal',
        'size'   : 75}
matplotlib.rc('font', **font)


algo_names = ['active', 'random']
main_curve = 'avg' # 'median'
error_type = 'std' # 'sem' #80
path_to_results = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/compare2' + '/'


gep_colors = [[0.3010,0.745,0.933],[0,0.447,0.7410],[222/255,21/255,21/255],[0.635,0.078,0.184],[0.466,0.674,0.188]]
matlab_colors = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.494,0.1844,0.556],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
matlab_colors2 = [[0.494,0.1844,0.556],[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.3010,0.745,0.933],[0.635,0.078,0.184]]
colors = matlab_colors

max_algo_n_eps = 0
algos_success = []
algo_all_results = []
algo_c = []
algo_cp = []
for algo in algo_names:
    path_to_algo = path_to_results + algo
    trials = os.listdir(path_to_algo)
    n_trials = 0
    max_n_eps = 0
    trial_success = []
    c_algo = []
    cp_algo = []
    for t in trials:
        path_to_trial = path_to_algo + '/' + t
        try:

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

            env = gym.make('MultiTaskFetchArm7-v0')#params['env_name'])
            if structure == 'modular' or structure == 'multiple_experts':
                n_tasks = env.unwrapped.n_tasks
            else:
                n_tasks = 1
            env.close()
            goal_replay = params['goal_replay']
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
            if len(episodes) > max_n_eps:
                max_n_eps = len(episodes)
                eps = episodes
            # extract test success rate
            test_success = data['test/success_rate']
            trial_success.append(test_success)

            # extract competence and cp
            n_points = test_success.shape[0]
            cp = np.zeros([n_points, n_tasks])
            c = np.zeros([n_points, n_tasks])
            if structure != 'flat':
                for i in range(n_tasks):
                    c[:, i] = data['train/C_task' + str(i)]
                    cp[:, i] = data['train/CP_task' + str(i)]
                c_algo.append(c)
                cp_algo.append(cp)


            n_trials += 1
        except:
            print('trial', t, 'has failed')

    if max_n_eps > max_algo_n_eps:
        max_algo_n_eps = max_n_eps
        algo_eps = eps

    # average and plot
    results_algo = np.zeros([n_trials, max_n_eps])
    results_algo.fill(np.nan)
    for i in range(n_trials):
        results_algo[i, :len(trial_success[i])] = np.array(trial_success[i])

    # convert c and cp to array
    cp_algo = np.array(cp_algo)
    c_algo = np.array(c_algo)

    if main_curve == 'avg':
        main_result = np.nanmean(results_algo, axis=0)
        # main_c = np.nanmean(c_algo, axis=0)
        # main_cp = np.nanmean(cp_algo, axis=0)
    elif main_curve == 'median':
        main_result = np.nanmedian(results_algo, axis=0)
        # main_c = np.nanmedian(c_algo, axis=0)
        # main_cp = np.nanmedian(cp_algo, axis=0)

    if error_type == 'std':
        sub = main_result - np.nanstd(results_algo, axis=0)
        sup = main_result + np.nanstd(results_algo, axis=0)
        # sub_c = main_c - np.nanstd(c_algo, axis=0)
        # sub_cp = main_cp - np.nanstd(cp_algo, axis=0)
        # sup_c = main_c + np.nanstd(c_algo, axis=0)
        # sup_cp = main_cp + np.nanstd(cp_algo, axis=0)
    elif error_type == 'sem':
        sub = main_result - np.nanstd(results_algo, axis=0) / np.sqrt(n_trials)
        sup = main_result + np.nanstd(results_algo, axis=0) / np.sqrt(n_trials)
        # sub_c = main_c - np.nanstd(c_algo, axis=0) / np.sqrt(n_trials)
        # sub_cp = main_cp - np.nanstd(cp_algo, axis=0) / np.sqrt(n_trials)
        # sup_c = main_c + np.nanstd(c_algo, axis=0) / np.sqrt(n_trials)
        # sup_cp = main_cp + np.nanstd(cp_algo, axis=0) / np.sqrt(n_trials)
    else:
        sub = np.nanpercentile(a=results_algo, q=1-error_type/100, axis=0)
        sup = np.nanpercentile(a=results_algo, q=error_type/100, axis=0)
        # sub_c = np.nanpercentile(a=c_algo, q=1-error_type/100, axis=0)
        # sub_cp = np.nanpercentile(a=cp_algo, q=1-error_type/100, axis=0)
        # sup_c = np.nanpercentile(a=c_algo, q=error_type/100, axis=0)
        # sup_cp = np.nanpercentile(a=cp_algo, q=error_type/100, axis=0)

    algo_all_results.append(results_algo)
    algos_success.append([main_result, sub, sup])
    # algo_c.append([main_c, sub_c, sup_c])
    # algo_cp.append([main_cp, sub_cp, sup_cp])


# compute t-test Welch
n_points = min(algo_all_results[0].shape[1], algo_all_results[1].shape[1])
t = np.zeros([n_points])
p = np.zeros([n_points])
for i in range(n_points):
    # t[i], p[i] = ttest_ind(algo_all_results[0][:, i], algo_all_results[1][:, i], nan_policy='propagate', equal_var=False)
    t[i], p[i] = wilcoxon(algo_all_results[0][:, i], algo_all_results[1][:, i])

# obtain one-tail p-value
p /= 2


# group all plots in same array
all_success = np.zeros([len(algo_names), max_algo_n_eps, 3])
all_success.fill(np.nan)
all_sub = np.zeros([len(algo_names), max_algo_n_eps])
all_sub.fill(np.nan)
all_sup = np.zeros([len(algo_names), max_algo_n_eps])
all_sup.fill(np.nan)

for i in range(len(algo_names)):
    for j in range(3):
        all_success[i, :len(algos_success[i][0]), j] = np.array(algos_success[i][j])

# downsample
factor = 4
ind = np.arange(0, n_points, factor)
algo_eps = algo_eps[ind]
all_success = all_success[:, ind, :]
p = p[ind]
sign_ind = np.argwhere(p < 0.05)


# plots
fig=plt.figure(figsize=(40,15),frameon=False) #40,15
ax = fig.add_subplot(111)
ax.spines['top'].set_linewidth(5)
ax.spines['right'].set_linewidth(5)
ax.spines['bottom'].set_linewidth(5)
ax.spines['left'].set_linewidth(5)
ax.tick_params(width=4, direction='in', length=10, labelsize='small')
for i in range(len(algo_names)):
    plt.plot(algo_eps, all_success[i, :, 0], linewidth=10, c=colors[i])
    plt.fill_between(algo_eps, all_success[i, :, 1], all_success[i, :, 2], alpha=0.2, facecolor=colors[i])
plt.yticks([0.25,0.50,0.75,1])
lab = plt.xlabel('Episodes (x$10^3$)')
# plt.ylim([-0.01, 1.11])
# plt.xlim([0, 300])
plt.scatter(algo_eps[sign_ind], 1.09 * np.ones([sign_ind.size]), linewidth=8, c='k', linestyle='--')
lab2 = plt.ylabel('Success rate')
leg = plt.legend(algo_names, frameon=False, fontsize=75, bbox_to_anchor=(0.5,-0.036,0.5,0.1)) #bbox_to_anchor=(0.49,-0.036,0.2,0.1))#
plt.savefig(path_to_results + 'plot_test_success_rate.svg',format='svg',  bbox_extra_artists=(leg,lab,lab2), bbox_inches='tight', dpi=200)
plt.savefig(path_to_results + 'plot_test_success_rate.png',  bbox_extra_artists=(leg,lab,lab2), bbox_inches='tight', dpi=200)


# # plot c, cp for M-ATR
# fig=plt.figure(figsize=(22,15),frameon=False)
# ax = fig.add_subplot(111)
# ax.spines['top'].set_linewidth(4)
# ax.spines['right'].set_linewidth(4)
# ax.spines['bottom'].set_linewidth(4)
# ax.spines['left'].set_linewidth(4)
# ax.tick_params(width=4, direction='in', length=10, labelsize='small')
# random.seed(10)
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# pl = []
# for i in range(n_tasks):
#     p = plt.plot(algo_eps, algo_c[0][0][:,i], linewidth=10, c=colors[i])
#     plt.fill_between(algo_eps, algo_c[0][1][:,i], algo_c[0][2][:,i], alpha=0.2, facecolor=colors[i])
#     pl.append(p)
# pl = [pl[i][0] for i in range(n_tasks)]
# plt.yticks([0.25,0.50,0.75,1])
# lab = plt.xlabel('episodes (x$10^3$)')
# plt.ylim([-0.005, 1.01])
# plt.xlim([0, 300])
# lab2 = plt.ylabel('Competence')
# leg = plt.legend(pl[:4] + [pl[6]], ['task ' + str(i + 1) for i in range(4)] + ['task 5-6-7'], loc=4, fontsize=70, frameon=False)
# plt.savefig(path_to_results + 'plot_c_matr.png', bbox_extra_artists=(leg,lab,lab2), bbox_inches='tight')
#
#
# fig=plt.figure(figsize=(22,15),frameon=False)
# ax = fig.add_subplot(111)
# ax.spines['top'].set_linewidth(4)
# ax.spines['right'].set_linewidth(4)
# ax.spines['bottom'].set_linewidth(4)
# ax.spines['left'].set_linewidth(4)
# ax.tick_params(width=4, direction='in', length=10, labelsize='small')
# random.seed(10)
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# pl = []
# for i in range(n_tasks):
#     p = plt.plot(algo_eps, algo_cp[0][0][:,i], linewidth=10, c=colors[i])
#     plt.fill_between(algo_eps, algo_cp[0][1][:,i], algo_cp[0][2][:,i], alpha=0.2, facecolor=colors[i])
#     pl.append(p)
# pl = [pl[i][0] for i in range(n_tasks)]
# plt.yticks([0.25,0.50,0.75,1])
# lab = plt.xlabel('episodes (x$10^3$)')
# plt.ylim([-0.005, 0.3])
# plt.xlim([0, 300])
# lab2 = plt.ylabel('Competence')
# leg = plt.legend(pl[:4] + [pl[6]], ['task ' + str(i + 1) for i in range(4)] + ['task 5-6-7'], loc=4, fontsize=70, frameon=False)
# plt.savefig(path_to_results + 'plot_cp_matr.png', bbox_extra_artists=(leg,lab,lab2), bbox_inches='tight')












