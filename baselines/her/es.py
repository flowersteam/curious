import numpy as np
from collections import deque
from baselines import logger


class SimpleES():
    def __init__(self, dim_in, tasks_id, nb_tasks, sigma=0.05, lmbda=38, nb_best=8, nb_init=500):
        """
        Looks for parameters of a linear function, mapping initial state to goal space. One mapping per goal space (task)

        """

        self.nb_tasks = nb_tasks
        self.tasks_id = tasks_id
        self.sigma = sigma
        self.lmbda = lmbda
        self.dim_in = dim_in
        self.dim_middle = 10
        self.dims_out = []
        self.N = []
        self.nb_biases = []
        self.nb_params = []
        for tasks in tasks_id:
            self.dims_out.append(len(tasks))
        for dim_out in self.dims_out:
            self.N.append(dim_in * self.dim_middle + self.dim_middle * dim_out)
            self.nb_biases.append(self.dim_middle + dim_out)
            self.nb_params.append(self.N[-1] + self.nb_biases[-1])

        self.nb_init = nb_init
        self.nb_best = nb_best

        self.means = None
        self.counts = None
        self.params = None


    def reset_search(self):
        # initialize the search
        self.means = [np.zeros([self.nb_params[i]]) for i in range(self.nb_tasks)] # track mean parameter (current best guess)
        self.counts_since = [0] * self.nb_tasks # counts number of samples since last update
        self.counts = [0] * self.nb_tasks # counts number of samples since last update
        self.memories_fitness = [deque(maxlen=self.lmbda)for _ in range(self.nb_tasks)]
        self.memories_params = [deque(maxlen=self.lmbda)for _ in range(self.nb_tasks)]
        self.memories_init_fitness = [deque(maxlen=self.nb_init) for _ in range(self.nb_tasks)]
        self.memories_init_params = [deque(maxlen=self.nb_init)for _ in range(self.nb_tasks)]



    def sample_goal(self, initial_state, probabilities, competences):

        assert probabilities.shape[0] == self.nb_tasks
        self.task = np.random.choice(range(self.nb_tasks), p=probabilities)

        dim_goal = len(self.tasks_id[self.task])
        self.random = False # false mean we used a set of parameter to produce the goal

        # draw random goals when the task is below half mastered
        if competences[self.task] < 0.001 or np.random.random() < 0.2:
            goal = np.random.uniform(-1, 1, dim_goal)
            self.params = []
            self.random = True

        # when competence is high enough, first draw a bunch of parameters at random
        elif self.counts[self.task] < self.nb_init:
            # Gloriot-Bengio initialization, allow uniform distribution in output, when uniform [-1,1] in input.
            self.params = np.random.uniform(- np.sqrt(6 / (self.dim_in + 1)), np.sqrt(6 / (self.dim_in + 1)), self.nb_params[self.task])
            goal = self.draw_goal(initial_state, self.params)

        # draw goals with current parameters
        else:
            self.params = np.random.normal(self.means[self.task], self.sigma, self.nb_params[self.task])
            goal = self.draw_goal(initial_state, self.params)
        return self.task.copy(), goal, self.params.copy(), self.random


    def update(self, fitness, params, random, task):

        # update using last set of parameters if the last goal was not produced at random
        if not random:
            if self.counts[task] < self.nb_init:
                self.memories_init_fitness[task].append(fitness)
                self.memories_init_params[task].append(params)
                self.counts[task] += 1
                # After initialization, pick lambda best and compute starting mean of parameters
                if self.counts[task] == self.nb_init:
                    # take n_bests fitnesses and average their parameters
                    ind_best = np.argsort(self.memories_init_fitness[task])[-self.nb_best:].squeeze()
                    best_params = np.array(self.memories_init_params[task])[ind_best, :]
                    self.means[task] = best_params.mean(axis=0)
                    self.counts[task] += 1
                    logger.info('Now using goal sampler for task ', task)
            else:
                self.memories_fitness[task].append(fitness)
                self.memories_params[task].append(params)
                self.counts_since[task] += 1
                if self.counts_since[task] == self.lmbda:
                    # reset counts since last update
                    self.counts_since[task] = 0
                    ind_best = np.argsort(np.array(self.memories_fitness[task]))[-self.nb_best:].squeeze()
                    best_params = np.array(self.memories_params[task])[ind_best, :]
                    self.means[task] = best_params.mean(axis=0)
                    logger.info('Mod', task)
                    logger.info('Mean params', self.means[task])




    def draw_goal(self, initial_state, params):

        weights = params[:self.N[self.task]].reshape([self.dim_in, self.dims_out[self.task]])
        biases = params[self.N[self.task]:]
        assert biases.shape[0] == self.nb_biases[self.task]

        goal = np.tanh(np.matmul(initial_state, weights) + biases)
        return goal


def return_cmaes_stuff(mu, N):
    if type(N) is int:
        N = np.array([N])
    weights = np.log(mu + 1 / 2) - np.log(np.asarray(range(1, mu + 1))).astype(np.float32)
    weights = weights / np.sum(weights)
    mueff = (np.sum(weights) ** 2) / np.sum(weights ** 2)

    cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = np.minimum(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
    damps = 1 + 2 * np.maximum(0, ((mueff - 1) / (N + 1)) ** 0.5 - 1) + cs
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    return [weights for _ in range(N.shape[0])], [mueff for _ in range(N.shape[0])], cc.tolist(), cs.tolist(), c1.tolist(), cmu.tolist(), damps.tolist(), chiN.tolist()

def update_cmaes(cs, ps, mueff, weights, xmean, xold, sigma, invsqrtC, lambda_, chiN, N, cc, pc, best_params, c1, cmu, C, damps, counteval, eigeneval, B, D):
    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
    hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * counteval / lambda_)) / chiN < 1.4 + 2 / (N + 1)
    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
    artmp = (1 / sigma) * (best_params - xold)
    C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T.dot(
        np.diag(weights)).dot(artmp)
    sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

    if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:
        eigeneval = counteval
        C = np.triu(C) + np.triu(C, 1).T
        D, B = np.linalg.eig(C)
        D = np.sqrt(D)
        invsqrtC = B.dot(np.diag(D ** -1).dot(B.T))

    return ps, pc, C, sigma, invsqrtC, B, D

class CMAES():
    def __init__(self, dim_in, tasks_id, nb_tasks, nb_cpu, sigma=0.05, lmbda=10, mu=3):
        """
        Looks for parameters of a linear function, mapping initial state to goal space. One mapping per goal space (task)

        """

        self.nb_tasks = nb_tasks
        self.tasks_id = tasks_id
        self.sigma = sigma
        self.lmbda = lmbda
        self.dim_in = dim_in
        self.nb_cpu = nb_cpu
        self.dim_middle = 10
        self.dims_out = []
        self.N = []
        for tasks in tasks_id:
            self.dims_out.append(len(tasks))
        for dim_out in self.dims_out:
            self.N.append(dim_in * self.dim_middle + self.dim_middle * dim_out + self.dim_middle + dim_out)

        self.mu = mu
        self.sigma = [sigma for _ in range(self.nb_tasks)]
        self.means = None
        self.params = None
        self.cma_weights, self.cma_mueff, self.cc, self.cs, self.c1, self.cmu, self.damps, self.chiN = return_cmaes_stuff(mu, np.array(self.N))

    def reset_search(self):
        # initialize the search
        self.means = [self.sigma[i] * np.random.randn(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]  # track mean parameter (current best guess)
        self.params = [np.zeros([self.N[i]]) for i in range(self.nb_tasks)]
        self.counts_since = [0] * self.nb_tasks # counts number of samples since last update
        self.memories_fitness = [deque(maxlen=self.lmbda)for _ in range(self.nb_tasks)]
        self.memories_params = [deque(maxlen=self.lmbda)for _ in range(self.nb_tasks)]

        self.counteval = [0] * self.nb_tasks
        self.pc = [np.zeros(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.ps =  [np.zeros(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.B = [np.eye(self.N[i], self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.D = [np.ones(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.C = [ self.B[i] * np.diag(self.D[i] ** 2) * self.B[i].T for i in range(self.nb_tasks)]
        self.invsqrtC = [self.B[i] * np.diag(self.D[i] ** -1) * self.B[i].T for i in range(self.nb_tasks)]
        self.eigeneval = [0 for _ in range(self.nb_tasks)]
        self.generation = [0 for _ in range(self.nb_tasks)]
     

    def sample_goal(self, initial_state, task):

        self.task = task
        task_return = [self.task for _ in range(self.nb_cpu)]

        # draw goals with current parameters

        self.params[self.task] = self.means[self.task] + self.sigma[self.task] * self.B[self.task].dot(self.D[self.task] * np.random.randn(self.N[self.task]).astype(np.float32))
        goal = [self.draw_goal(initial_state[i], self.params[self.task], self.task) for i in range(self.nb_cpu)]

        return task_return, goal, self.params[self.task].copy()


    def update(self, fitness_all, params_all, randoms, tasks):
        nb_rollouts = len(params_all)
        inds = []
        for i in range(nb_rollouts):
            inds.append(np.arange(i, self.nb_cpu * nb_rollouts, nb_rollouts))
        for i in range(nb_rollouts):
            fitness = np.array([fitness_all[j] for j in inds[i]]).mean()
            task = tasks[i]
            random = randoms[i]
            params = params_all[i]
            # update using last set of parameters if the last goal was not produced at random
            if not random:
                self.memories_fitness[task].append(fitness)
                self.memories_params[task].append(params)
                self.counts_since[task] += 1
                self.counteval[task] += 1
                if self.counts_since[task] == self.lmbda:
                    # reset counts since last update
                    self.counts_since[task] = 0
                    x_old = self.means[task].copy()
                    av_cumulative_rew = np.array(self.memories_fitness[task]).mean()
                    ind_best = np.argsort(np.array(self.memories_fitness[task]))[-self.mu:].squeeze()
                    best_params = np.array(self.memories_params[task])[ind_best, :]
                    self.means[task] = self.cma_weights[task].dot(best_params.copy())
                    self.ps[task], self.pc[task], self.C[task], \
                    self.sigma[task], self.invsqrtC[task], \
                    self.B[task], self.D[task] = update_cmaes(self.cs[task], self.ps[task],
                                                            self.cma_mueff[task], self.cma_weights[task],
                                                            self.means[task], x_old, self.sigma[task],
                                                            self.invsqrtC[task], self.lmbda,
                                                            self.chiN[task], self.N[task],
                                                            self.cc[task], self.pc[task],
                                                            best_params, self.c1[task],
                                                            self.cmu[task], self.C[task],
                                                            self.damps[task], self.counteval[task],
                                                            self.eigeneval[task], self.B[task], self.D[task])


                    self.generation[task] += 1
                    logger.info('Mod', task, 'av cumulative reward of gen #',self.generation[task], ': ', av_cumulative_rew)




    def draw_goal(self, initial_state, params, task):
        # first layer
        ind = 0
        weights = params[ind: ind + self.dim_in * self.dim_middle].reshape([self.dim_in, self.dim_middle])
        ind += self.dim_in * self.dim_middle
        biases = params[ind: ind + self.dim_middle]
        ind += self.dim_middle
        tmp = np.tanh(np.matmul(initial_state, weights) + biases)

        # second layer
        weights = params[ind: ind + self.dim_middle * self.dims_out[task]].reshape([self.dim_middle, self.dims_out[task]])
        ind += self.dim_middle * self.dims_out[task]
        biases = params[ind: ind + self.dims_out[task]]
        ind += self.dims_out[task]
        assert ind == params.shape[0]

        goal = np.tanh(np.matmul(tmp, weights) + biases)

        return goal


class CMAES_distrib():
    def __init__(self, dim_in, tasks_id, nb_tasks, nb_cpu, sigma=0.05, lmbda=10, mu=3):
        """
        Looks for parameters of a linear function, mapping initial state to goal space. One mapping per goal space (task)

        """

        self.nb_tasks = nb_tasks
        self.tasks_id = tasks_id
        self.sigma = sigma
        self.lmbda = lmbda
        self.dim_in = dim_in
        self.nb_cpu = nb_cpu
        self.dim_middle = 10
        self.dim_out = 4
        self.N = [dim_in * self.dim_middle + self.dim_middle * self.dim_out + self.dim_middle + self.dim_out for _ in range(self.nb_tasks)]

        self.mu = mu
        self.sigma = [sigma for _ in range(self.nb_tasks)]
        self.means = None
        self.params = None
        self.cma_weights, self.cma_mueff, self.cc, self.cs, self.c1, self.cmu, self.damps, self.chiN = return_cmaes_stuff(mu, np.array(self.N))

    def reset_search(self):
        # initialize the search
        self.means = [self.sigma[i] * np.random.randn(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]  # track mean parameter (current best guess)
        self.params = [np.zeros([self.N[i]]) for i in range(self.nb_tasks)]
        self.counts_since = [0] * self.nb_tasks  # counts number of samples since last update
        self.memories_fitness = [deque(maxlen=self.lmbda) for _ in range(self.nb_tasks)]
        self.memories_params = [deque(maxlen=self.lmbda) for _ in range(self.nb_tasks)]

        self.counteval = [0] * self.nb_tasks
        self.pc = [np.zeros(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.ps = [np.zeros(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.B = [np.eye(self.N[i], self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.D = [np.ones(self.N[i]).astype(np.float32) for i in range(self.nb_tasks)]
        self.C = [self.B[i] * np.diag(self.D[i] ** 2) * self.B[i].T for i in range(self.nb_tasks)]
        self.invsqrtC = [self.B[i] * np.diag(self.D[i] ** -1) * self.B[i].T for i in range(self.nb_tasks)]
        self.eigeneval = [0 for _ in range(self.nb_tasks)]
        self.generation = [0 for _ in range(self.nb_tasks)]

    def sample_goal(self, initial_state, task):

        self.task = task
        task_return = [self.task for _ in range(self.nb_cpu)]

        # draw goals with current parameters

        self.params[self.task] = self.means[self.task] + self.sigma[self.task] * self.B[self.task].dot(self.D[self.task] * np.random.randn(self.N[self.task]).astype(np.float32))
        goal = [self.draw_goal(initial_state[i], self.params[self.task], self.task) for i in range(self.nb_cpu)]

        return task_return, goal, self.params[self.task].copy()

    def update(self, fitness_all, params_all, randoms, tasks):
        nb_rollouts = len(params_all)
        inds = []
        for i in range(nb_rollouts):
            inds.append(np.arange(i, self.nb_cpu * nb_rollouts, nb_rollouts))
        for i in range(nb_rollouts):
            fitness = np.array([fitness_all[j] for j in inds[i]]).mean()
            task = tasks[i]
            random = randoms[i]
            params = params_all[i]
            # update using last set of parameters if the last goal was not produced at random
            if not random:
                self.memories_fitness[task].append(fitness)
                self.memories_params[task].append(params)
                self.counts_since[task] += 1
                self.counteval[task] += 1
                if self.counts_since[task] == self.lmbda:
                    # reset counts since last update
                    self.counts_since[task] = 0
                    x_old = self.means[task].copy()
                    av_cumulative_rew = np.array(self.memories_fitness[task]).mean()
                    ind_best = np.argsort(np.array(self.memories_fitness[task]))[-self.mu:].squeeze()
                    best_params = np.array(self.memories_params[task])[ind_best, :]
                    self.means[task] = self.cma_weights[task].dot(best_params.copy())
                    self.ps[task], self.pc[task], self.C[task], \
                    self.sigma[task], self.invsqrtC[task], \
                    self.B[task], self.D[task] = update_cmaes(self.cs[task], self.ps[task],
                                                            self.cma_mueff[task], self.cma_weights[task],
                                                            self.means[task], x_old, self.sigma[task],
                                                            self.invsqrtC[task], self.lmbda,
                                                            self.chiN[task], self.N[task],
                                                            self.cc[task], self.pc[task],
                                                            best_params, self.c1[task],
                                                            self.cmu[task], self.C[task],
                                                            self.damps[task], self.counteval[task],
                                                            self.eigeneval[task], self.B[task], self.D[task])

                    self.generation[task] += 1
                    logger.info('Mod', task, 'av cumulative reward of gen #', self.generation[task], ': ', av_cumulative_rew)

    def draw_goal(self, initial_state, params, task):
        # first layer
        ind = 0
        weights = params[ind: ind + self.dim_in * self.dim_middle].reshape([self.dim_in, self.dim_middle])
        ind += self.dim_in * self.dim_middle
        biases = params[ind: ind + self.dim_middle]
        ind += self.dim_middle
        tmp = np.tanh(np.matmul(initial_state, weights) + biases)

        # second layer
        weights = params[ind: ind + self.dim_middle * self.dim_out].reshape([self.dim_middle, self.dim_out])
        ind += self.dim_middle * self.dim_out
        biases = params[ind: ind + self.dim_out]
        ind += self.dim_out
        assert ind == params.shape[0]

        params = np.tanh(np.matmul(tmp, weights) + biases)
        goal = np.tanh(np.random.normal(loc=params[0:2], scale=np.abs(params[2:4])))

        # if task == 0:
        #     loc = initial_state[7:9]
        #     goal = np.random.normal(loc=loc, scale=np.array([0.2,0.2])).clip(-1,1)
        # elif task == 1:
        #     loc = initial_state[5:7]
        #     goal = np.random.normal(loc=loc, scale=np.array([0.2,0.2])).clip(-1,1)
        # else:
        #     goal = np.random.uniform(-1,1,2)


        return goal

