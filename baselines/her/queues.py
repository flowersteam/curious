from collections import deque
import itertools
import numpy as np
from scipy.signal import savgol_filter
from pandas import ewma

class CompetenceQueue():
    def __init__(self, window = 100):
        self.window = window
        self.successes = deque(maxlen=2 * self.window)
        self.CP = 0.
        self.C = 0.

    def update(self, success_list):
        for success in success_list:
            self.successes.append(success)

        if self.size > 2:
            window = min(self.size // 2, self.window)
            q1 = list(itertools.islice(self.successes, self.size - window, self.size))
            q2 = list(itertools.islice(self.successes, self.size - 2 * window, self.size - window))
            self.CP = np.abs(np.sum(q1) - np.sum(q2)) / (2 * window)
            self.C = np.sum(q1) / window

    @property
    def size(self):
        return len(self.successes)

    @property
    def full(self):
        return self.size == self.successes.maxlen

    def clear_queue(self):
        self.successes = deque(maxlen=2 * self.window)
        self.CP = 0
        self.C = 0.
