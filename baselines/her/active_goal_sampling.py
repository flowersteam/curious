import numpy as np
from gym.spaces import Box
from collections import deque


# Implementation of SAGG-RIAC

class SAGG_RIAC():
    def __init__(self, min, max, continuous_competence=False):

        assert len(min) == len(max)
        self.maxlen = 200
        self.regions = [[deque(maxlen=self.maxlen + 1), deque(maxlen=self.maxlen + 1)]]
        self.region_bounds = [Box(min, max, dtype=np.float32)]
        self.interest = [0.]
        self.probas = [1.]
        self.nb_dims = len(min)
        self.window_cp = 100
        self.temperature = 20
        self.nb_split_attempts = 50
        self.continuous_competence = continuous_competence
        self.max_difference = 0.3
        self.init_size = max - min

    def update(self, goals, binary_competence, continuous_competence=None):
        if len(goals) > 0:
            new_split = False
            all_order = None
            regions = [None] * len(goals)
            for i, goal in enumerate(goals):
                for j, rb in enumerate(self.region_bounds):
                    if rb.contains(goal):
                        regions[i] = j
                        break

            if self.continuous_competence:
                c_or_cps = continuous_competence
            else:
                c_or_cps = binary_competence

            # add new outcomes and goals to regions
            for reg, c_or_cp, goal in zip(regions, c_or_cps, goals):
                self.regions[reg][0].append(c_or_cp)
                self.regions[reg][1].append(goal)

            # check if need to split
            ind_split = []
            new_bounds = []
            new_sub_regions = []
            for reg in range(self.nb_regions):
                if len(self.regions[reg][0]) > self.maxlen:
                    # try nb_split_attempts splits
                    best_split_score = 0
                    best_abs_interest_diff = 0
                    best_bounds = None
                    best_sub_regions = None
                    is_split = False
                    for i in range(self.nb_split_attempts):
                        sub_reg1 = [deque(), deque()]
                        sub_reg2 = [deque(), deque()]

                        # repeat until the two sub regions contain at least 1/10 of the mother region
                        while len(sub_reg1[0]) < self.maxlen / 4 or  len(sub_reg2[0]) < self.maxlen / 4:
                            # decide on dimension
                            dim = np.random.choice(range(self.nb_dims))
                            threshold = self.region_bounds[reg].sample()[dim]
                            bounds1 = Box(self.region_bounds[reg].low, self.region_bounds[reg].high, dtype=np.float32)
                            bounds1.high[dim] = threshold
                            bounds2 = Box(self.region_bounds[reg].low, self.region_bounds[reg].high, dtype=np.float32)
                            bounds2.low[dim] = threshold
                            bounds = [bounds1, bounds2]
                            valid_bounds = True
                            if np.any(bounds1.high - bounds1.low < self.init_size / 5):
                                valid_bounds = False
                            if np.any(bounds2.high - bounds2.low < self.init_size / 5):
                                valid_bounds = valid_bounds and False

                            # perform split in sub regions
                            sub_reg1 = [deque(), deque()]
                            sub_reg2 = [deque(), deque()]
                            for i, goal in enumerate(self.regions[reg][1]):
                                if bounds1.contains(goal):
                                    sub_reg1[1].append(goal)
                                    sub_reg1[0].append(self.regions[reg][0][i])
                                else:
                                    sub_reg2[1].append(goal)
                                    sub_reg2[0].append(self.regions[reg][0][i])
                            sub_regions = [sub_reg1, sub_reg2]

                        # compute interest
                        interest = np.zeros([2])
                        for i in range(2):
                            if self.continuous_competence:
                                cp_window = min(len(sub_regions[i][0]), self.window_cp)
                                cp = np.abs(np.array(sub_regions[i][0])[-cp_window].mean())
                            else:
                                cp_window = min(len(sub_regions[i][0]) // 2, self.window_cp)
                                cp = np.array(sub_regions[i][0])[- 2 * cp_window: - cp_window].mean() - np.array(sub_regions[i][0])[- cp_window:].mean()
                            interest[i] = np.abs(cp)

                        # compute score
                        split_score = len(sub_reg1) * len(sub_reg2) * np.abs(interest[0] - interest[1])
                        if split_score >= best_split_score and np.abs(interest[0] - interest[1]) >= self.max_difference / 2 and valid_bounds:
                            best_abs_interest_diff = np.abs(interest[0] - interest[1])
                            print(interest)

                            best_split_score = split_score
                            best_sub_regions = sub_regions
                            best_bounds = bounds
                            is_split = True
                            if interest[0] >= interest[1]:
                                order = [1, -1]
                            else:
                                order = [-1, 1]
                    if is_split:
                        ind_split.append(reg)
                        if best_abs_interest_diff > self.max_difference:
                            self.max_difference = best_abs_interest_diff
                    else:
                        self.regions[reg][0] = deque(np.array(self.regions[reg][0])[- int (3 * len(self.regions[reg][0]) / 4):], maxlen=self.maxlen + 1)
                        self.regions[reg][1] = deque(np.array(self.regions[reg][1])[- int(3 * len(self.regions[reg][1]) / 4):], maxlen=self.maxlen + 1)
                    new_bounds.append(best_bounds)
                    new_sub_regions.append(best_sub_regions)

            # implement splits
            for i, reg in enumerate(ind_split):
                all_order = [0] * self.nb_regions
                all_order.pop(reg)
                all_order.insert(reg, order[0])
                all_order.insert(reg, order[1])

                new_split = True
                self.region_bounds.pop(reg)
                self.region_bounds.insert(reg, new_bounds[i][0])
                self.region_bounds.insert(reg, new_bounds[i][1])

                self.regions.pop(reg)
                self.regions.insert(reg, new_sub_regions[i][0])
                self.regions.insert(reg, new_sub_regions[i][1])

                self.interest.pop(reg)
                self.interest.insert(reg, 0)
                self.interest.insert(reg, 0)

                self.probas.pop(reg)
                self.probas.insert(reg, 0)
                self.probas.insert(reg, 0)


            # recompute interest
            for i in range(self.nb_regions):
                if len(self.regions[i][0]) > 10:
                    if self.continuous_competence:
                        cp_window = min(len(self.regions[i][0]), self.window_cp)
                        cp = np.abs(np.array(self.regions[i][0])[-cp_window].mean())
                    else:
                        cp_window = min(len(self.regions[i][0]) // 2, self.window_cp)
                        cp = np.array(self.regions[i][0])[- 2 * cp_window: - cp_window].mean() - np.array(self.regions[i][0])[- cp_window:].mean()
                else:
                    cp = 0
                self.interest[i] = np.abs(cp)
                exp_int = np.exp(self.temperature * np.array(self.interest))
                probas = exp_int / exp_int.sum()
                self.probas = probas.tolist()

            assert len(self.probas) == len(self.regions)
            return new_split, all_order

        else:
            return False, None

    def sample_goal(self):

        # sample region
        if np.random.rand() < 0.2:
            region_id = np.random.choice(range(self.nb_regions))
        else:
            region_id = np.random.choice(range(self.nb_regions), p=np.array(self.probas))


        # sample goal
        goal = self.region_bounds[region_id].sample()

        return goal

    @property
    def nb_regions(self):
        return len(self.regions)

    @property
    def get_regions(self):
        return self.region_bounds


