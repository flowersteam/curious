import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import gym
import matplotlib as mpl
import gym_flowers
import pickle

path = '/home/flowers/Desktop/Scratch/modular_her_baseline/baselines/her/experiment/plafrim_results/ModularArm012-v0/53' +'/'

# extract params from json
with open(path + 'params.json') as json_file:
    params = json.load(json_file)

modular = True #params['modular']
modular_strategy = params['modular_strategy']
active_goal_selection = params['active_goal_selection']
env_name = params['env_name']
env = gym.make(env_name)
nb_timesteps = env.unwrapped.n_timesteps
if modular: n_modules = env.unwrapped.n_modules
env.close()
replay_strategy = params['replay_strategy']
n_cycles = params['n_cycles']
rollout_batch_size = params['rollout_batch_size']
n_cpu = 19 #params['num_cpu']
n_test_eps = params['n_test_rollouts']
# extract results
data = pd.read_csv(path+'progress.csv')
# extract goal and module selections
if 'active' in modular_strategy:
    with open(path + 'extra_saves.pk', 'rb') as f:
        dict_in = pickle.load(f)
    goals = dict_in['goals']
    modules = dict_in['modules']
    changes = dict_in['changes']
    first_change = dict_in['first_change']
    fitness = dict_in['fitness']
    if active_goal_selection:
        init_o = dict_in['init_obs']
        rand = dict_in['random']

# rand = np.random.choice([True, False], p=[0.2,0.8], size=goals.shape[0])
episodes = data['train/episode'] * n_cpu
steps = episodes * nb_timesteps

test_success = data['test/success_rate']
n_points = test_success.shape[0]

if modular_strategy == 'active':
    mod_choices = np.zeros([n_points, n_modules])
    proba = np.zeros([n_points, n_modules])
    cp = np.zeros([n_points, n_modules])
    c = np.zeros([n_points, n_modules])
    for i in range(n_modules):
        mod_choices[:, i] = data['train/%_mod' + str(i)]
        proba[:, i] = data['train/p_mod' + str(i)]
        cp[:, i] = data['train/CP_mod' + str(i)]
        c[:, i] = data['train/C_mod' + str(i)]

ind_comp = []
goals_valid = []
random_valid = []
init_o_valid = []
fitness_valid = []
changes_valid = []
ind_valid = []
ind_init = 100
for i in range(n_modules):
    print(i)
    try:
        # for each module, only take the corresponding goals
        ind_mod = np.argwhere(modules==i).squeeze()
        goals_valid.append(goals[ind_mod])
        random_valid.append(rand[(ind_mod/19).astype(np.int)])
        init_o_valid.append(init_o[ind_mod])
        fitness_valid.append(fitness[ind_mod])
        changes_valid.append(changes[ind_mod])
        # only takes goals sampled by goal sampler, from the 100th
        ind_valid.append(np.argwhere(random_valid[i] == False).squeeze())
        goals_valid[i] = goals_valid[i][ind_valid[i][ind_init:]]
        random_valid[i] = random_valid[i][ind_valid[i][ind_init:]]
        init_o_valid[i] = init_o_valid[i][ind_valid[i][ind_init:]]
        fitness_valid[i] = fitness_valid[i][ind_valid[i][ind_init:]]
        changes_valid[i] = changes_valid[i][ind_valid[i][ind_init:]]
    except:
        goals_valid.append([])
        random_valid.append([])
        init_o_valid.append([])
        fitness_valid.append([])
        changes_valid.append([])

color = np.arange(50,255)
# plot goals
for i in range(n_modules):
    try:
        fit = fitness_valid[i]

        plt.figure(frameon=False)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Distribution of goals for module ' + str(i))
        x=100
        x=goals_valid[i].shape[0]
        plt.scatter(goals_valid[i][-x:,0], goals_valid[i][-x:,1], s=2, c=fit[-x:])
        plt.colorbar()
    except:
        pass

ind_stick = [7, 8]
ind_object = [5, 6]
smoothness = 500

# for module 0, compute distance between goal and stick position
dist = np.linalg.norm(goals_valid[0] - init_o_valid[0][:, ind_stick], ord=2, axis=1)
plt.figure(frameon=False)
plt.plot(dist, 'o', markersize=2)
plt.ylim([0, np.sqrt(2*2**2)])
plt.title('Distance between goal for module 0 and stick')
plt.xlabel('# goals selected by goal selector for module 0')
plt.ylabel('Euclidean distance')

dist_av = np.zeros([dist.shape[0] - smoothness])
for i in range(dist_av.shape[0]):
    dist_av[i] = dist[i: i+smoothness].mean()
plt.figure(frameon=False)
plt.plot(dist_av, 'o', markersize=2)
plt.ylim([0, np.sqrt(2*2**2)])
plt.title('Distance between goal for module 0 and stick')
plt.xlabel('# goals selected by goal selector for module 0')
plt.ylabel('Euclidean distance')


try:
    dist = np.linalg.norm(goals_valid[1] - init_o_valid[1][:, ind_object], ord=2, axis=1)
    plt.figure(frameon=False)
    plt.plot(dist, 'o', markersize=2)
    plt.title('Distance between goal for module 1 and object')
    plt.xlabel('# goals selected by goal selector for module 1')
    plt.ylabel('Euclidean distance')
    plt.ylim([0, np.sqrt(2 * 2.5 ** 2)])

    dist_av = np.zeros([dist.shape[0] - smoothness])
    for i in range(dist_av.shape[0]):
        dist_av[i] = dist[i: i + smoothness].mean()
    plt.figure(frameon=False)
    plt.plot(dist_av, 'o', markersize=2)
    plt.title('Distance between goal for module 1 and object')
    plt.xlabel('# goals selected by goal selector for module 0')
    plt.ylim([0, np.sqrt(2 * 2.5 ** 2)])
    plt.ylabel('Euclidean distance')
except:
    pass


#
# fitness_av = np.zeros([fitness_valid[0].shape[0]-smoothness])
# for i in range(fitness_av.shape[0]):
#     fitness_av[i] = fitness_valid[0][i:i+smoothness].mean()
# plt.figure(frameon=False)
# plt.plot(fitness_av, 'o', markersize=2)
# plt.title('Fitness running average')
# plt.xlabel('# goals selected by goal selector for module 0')
# plt.ylabel('fitness')
#
# plt.figure(frameon=False)
# plt.plot(fitness_valid[0], 'o', markersize=2)
# plt.title('Fitness')
# plt.xlabel('# goals selected by goal selector for module 0')
# plt.ylabel('fitness')
#
# try:
#     fitness_av = np.zeros([fitness_valid[1].shape[0]-smoothness])
#     for i in range(fitness_av.shape[0]):
#         fitness_av[i] = fitness_valid[1][i:i+smoothness].mean()
#     plt.figure(frameon=False)
#     plt.plot(fitness_av, 'o', markersize=2)
#     plt.title('Fitness running average')
#     plt.xlabel('# goals selected by goal selector for module 1')
#     plt.ylabel('fitness')
#
#     plt.figure(frameon=False)
#     plt.plot(fitness_valid[1], 'o', markersize=2)
#     plt.title('Fitness')
#     plt.xlabel('# goals selected by goal selector for module 1')
#     plt.ylabel('fitness')
# except:
#     pass

plt.show()

stop=1

