import os

import numpy as np
from paddlehub.common.logger import logger

from slda_novel.util import rand, rand_k


class VoseAlias(object):
    """Vose's Alias Method.
    """

    def __init__(self):
        self.__alias = None
        self.__prob = None  # np.array

    def initialize(self, distribution):
        """Initialize the alias table according to the input distribution
        Arg:
            distribution: Numpy array.
        """
        size = distribution.shape[0]
        self.__alias = np.zeros(size, dtype=np.int64)
        self.__prob = np.zeros(size)
        sum_ = np.sum(distribution)
        p = distribution / sum_ * size  # Scale up probability.
        large, small = [], []
        for i, p_ in enumerate(p):
            if p_ < 1.0:
                small.append(i)
            else:
                large.append(i)

        while large and small:
            l = small[0]
            g = large[0]
            small.pop(0)
            large.pop(0)
            self.__prob[l] = p[l]
            self.__alias[l] = g
            p[g] = p[g] + p[l] - 1  # A more numerically stable option.
            if p[g] < 1.0:
                small.append(g)
            else:
                large.append(g)

        while large:
            g = large[0]
            large.pop(0)
            self.__prob[g] = 1.0

        while small:
            l = small[0]
            small.pop(0)
            self.__prob[l] = 1.0

    def generate(self):
        """Generate samples from given distribution.
        """
        dart1 = rand_k(self.size())
        dart2 = int(rand())
        return dart1 if dart2 > self.__prob[dart1] else self.__alias[dart1]

    def size(self):
        return self.__prob.shape[0]
