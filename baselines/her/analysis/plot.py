import os
import pickle
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gym
import gym_flowers

os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'

font = {'size'   : 75}
matplotlib.rc('font', **font)

matlab_colors2 = [[0,0.447,0.7410],[0.85,0.325,0.098],[0.466,0.674,0.188],[0.929,0.694,0.125],[0.494,0.1844,0.556],[0,0.447,0.7410],[0.3010,0.745,0.933],[0.85,0.325,0.098],
                  [0.466,0.674,0.188],[0.929,0.694,0.125],
                  [0.3010,0.745,0.933],[0.635,0.078,0.184]]
colors = matlab_colors2

folder_path = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/MultiTaskFetchArm8-v5/'
# folder_path = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/Rooms3-v0/'
trials = [445] #list(range(30,40))#
for trial in trials:
    print('Ploting trial', trial)
    path = folder_path + str(trial) + '/'

    # extract params from json
    with open(path + 'params.json') as json_file:
        params = json.load(json_file)

    structure = params['structure']
    if structure == 'flat':
        task_selection = None
        goal_selection = 'random'
    else:
        task_selection = params['task_selection']
        goal_selection = params['goal_selection']


    env = gym.make(params['env_name'])
    if structure in ['curious', 'task_experts']:
        n_tasks = env.unwrapped.n_tasks
    else:
        n_tasks = 1
    env.close()

    goal_replay = params['goal_replay']
    n_cycles = params['n_cycles']
    rollout_batch_size = params['rollout_batch_size']
    n_cpu = params['num_cpu']

    # extract results
    data = pd.read_csv(path+'progress.csv')
    episodes = data['train/episode']
    n_epochs = len(episodes)
    n_eps = n_cpu * rollout_batch_size * n_cycles
    episodes = np.arange(n_eps, n_epochs * n_eps + 1, n_eps)

    test_success = data['test/success_rate']
    n_points = test_success.shape[0]

    if structure=='curious' or structure=='modular':
        task_choices = np.zeros([n_points, n_tasks])
        proba = np.zeros([n_points, n_tasks])
        cp = np.zeros([n_points, n_tasks])
        c = np.zeros([n_points, n_tasks])
        eval_c = np.zeros([n_points, n_tasks])
        for i in range(n_tasks):
            task_choices[:, i] = data['train/%_task' + str(i)]
            proba[:, i] = data['train/p_task' + str(i)]
            cp[:, i] = data['train/CP_task' + str(i)]
            c[:, i] = data['train/C_task' + str(i)]
            eval_c[:, i] = data['test/C_task' + str(i)]

    elif structure == 'task_experts':
        task_choices = np.zeros([n_points, n_tasks])
        proba = np.zeros([n_points, n_tasks])
        cp = np.zeros([n_points, n_tasks])
        c = np.zeros([n_points, n_tasks])
        eval_c = np.zeros([n_points, n_tasks])
        last_p=[0] * n_tasks
        for p in range(n_points):
            current_i = data['IND_TASK_rollout'][p]
            for i in range(n_tasks):
                if i == current_i:
                    current_p = p
                    last_p[i] = current_p
                else:
                    current_p = last_p[i]

                task_choices[p, i] = data['train/%_task' + str(i)][current_p]
                proba[p, i] = data['train/p_task' + str(i)][current_p]
                cp[p, i] = data['train/CP_task' + str(i)][current_p]
                c[p, i] = data['train/C_task' + str(i)][current_p]
                eval_c[p, i] = data['test/C_task' + str(i)][current_p]

    episodes = episodes / 1000


    # plot success rates
    fig=plt.figure(figsize=(22,15),frameon=False)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.tick_params(width=4, direction='in', length=10, labelsize='small')
    plt.plot(episodes, test_success, linewidth=5)
    plt.yticks([0.25,0.50,0.75,1])
    lab = plt.xlabel('Episodes (x$10^3$)')
    plt.ylim([-0.005, 1.01])
    lab2 = plt.ylabel('Test success rate')
    plt.savefig(path + 'plot_test_success_rate.png', bbox_extra_artists=(lab,lab2), bbox_inches='tight', dpi=300)

    if structure=='curious'  or structure=='modular' or structure=='task_experts':
        # plot evolution of task selection
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(n_tasks):
            plt.plot(episodes, task_choices[:, i], linewidth=10, c=colors[i])
        leg = plt.legend(['task ' + str(i+1) for i in range(n_tasks)])
        lab = plt.xlabel('episodes (x$10^3$)')
        plt.ylim([-0.001, 1.01])
        lab2 = plt.ylabel('Task selection')
        plt.savefig(path + 'plot_task_selection.png', bbox_extra_artists=(lab,lab2,leg), bbox_inches='tight', dpi=300)

        # plot evolution of competence progress
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(n_tasks):
            plt.plot(episodes, cp[:,i], linewidth=10, c=colors[i])
        plt.ylim([-0.0025,cp.max()+0.01])
        leg = plt.legend(['task ' + str(i + 1) for i in range(n_tasks)])
        lab = plt.xlabel('Episodes (x$10^3$)')
        lab2 = plt.ylabel('LP')
        plt.savefig(path + 'plot_cp.png', bbox_extra_artists=(lab,lab2,leg), bbox_inches='tight', dpi=300)

        # plot evolution of competence
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(n_tasks):
            p = plt.plot(episodes, c[:,i], linewidth=10, c=colors[i])
        leg = plt.legend(['task ' + str(i + 1) for i in range(n_tasks)])
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([-0.01, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('Competence')
        plt.savefig(path + 'plot_c.png', bbox_extra_artists=(lab,lab2,leg), bbox_inches='tight', dpi=300) # add leg

        # plot evolution of competence
        fig = plt.figure(figsize=(22, 15), frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(n_tasks):
            plt.plot(episodes, eval_c[:, i], linewidth=10, c=colors[i])
        leg = plt.legend(['task ' + str(i + 1) for i in range(n_tasks)])
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([-0.01, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('Evaluation Competence')
        plt.savefig(path + 'plot_c_eval.png', bbox_extra_artists=(lab, lab2, leg), bbox_inches='tight', dpi=300)


        # plot evolution of proba to replay task
        plt.figure(figsize=(22,15),frameon=False)
        ax = fig.add_subplot(111)
        ax.spines['top'].set_linewidth(4)
        ax.spines['right'].set_linewidth(4)
        ax.spines['bottom'].set_linewidth(4)
        ax.spines['left'].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=10, labelsize='small')
        for i in range(n_tasks):
            plt.plot(episodes, proba[:,i], linewidth=10, c=colors[i])
        leg = plt.legend(['task ' + str(i + 1) for i in range(n_tasks)])
        lab = plt.xlabel('Episodes (x$10^3$)')
        plt.ylim([0, 1.01])
        plt.yticks([0.25, 0.50, 0.75, 1])
        lab2 = plt.ylabel('probabilities')
        plt.savefig(path + 'plot_buffer_cp_proba.png', bbox_extra_artists=(lab,lab2,leg), bbox_inches='tight', dpi=300)
