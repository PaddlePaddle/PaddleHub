# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from multiprocessing.pool import ThreadPool
import copy
import json
import math
import numpy as np
import six
import time

from paddlehub.common.logger import logger
from paddlehub.common.utils import mkdir

if six.PY3:
    INF = math.inf
else:
    INF = float("inf")


class PSHE2(object):
    def __init__(
            self,
            evaluator,
            cudas=["0"],
            popsize=5,
            output_dir=None,
            alpha=0.5,
            epsilon=0.2,
    ):

        self._num_thread = len(cudas)
        self._popsize = popsize
        self._alpha = alpha
        self._epsilon = epsilon
        self._iteration = 0
        self.cudas = cudas
        self.is_cuda_free = {"free": [], "busy": []}
        self.is_cuda_free["free"] = cudas

        self.evaluator = evaluator
        self.init_input = evaluator.get_init_params()
        self.num_hparm = len(self.init_input)

        self.best_hparams_per_pop = [[0] * self.num_hparm] * self._popsize
        self.best_reward_per_pop = [INF] * self._popsize
        self.momentums = [[0] * self.num_hparm] * self._popsize
        self.best_hparms_all_pop = []
        self.best_reward_all_pop = INF
        self.current_hparams = [[0] * self.num_hparm] * self._popsize
        for i in range(self.popsize):
            self.current_hparams[i] = self.randomSolution()

        if output_dir is None:
            now = int(time.time())
            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(now))
            self._output_dir = "output_" + time_str
        else:
            self._output_dir = output_dir

    @property
    def thread(self):
        return self._num_thread

    @property
    def popsize(self):
        return self._popsize

    @property
    def alpha(self):
        return self._alpha

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def iteration(self):
        return self._iteration

    def set_output_dir(self, output_dir=None):
        if output_dir is not None:
            output_dir = output_dir
        else:
            output_dir = self._output_dir
        return output_dir

    def randomSolution(self):
        solut = [0] * self.num_hparm
        for i in range(self.num_hparm):
            ratio = (np.random.random_sample() - 0.5) * 2.0
            if ratio >= 0:
                solut[i] = (
                    1.0 - self.init_input[i]) * ratio + self.init_input[i]
            else:
                solut[i] = (
                    self.init_input[i] + 1.0) * ratio + self.init_input[i]
        return solut

    def smallPeturb(self):
        for i in range(self.popsize):
            for j in range(self.num_hparm):
                ratio = (np.random.random_sample() - 0.5) * 2.0
                if ratio >= 0:
                    self.current_hparams[i][j] = (
                        1.0 - self.current_hparams[i][j]
                    ) * ratio * self.epsilon + self.current_hparams[i][j]
                else:
                    self.current_hparams[i][j] = (
                        self.current_hparams[i][j] +
                        1.0) * ratio * self.epsilon + self.current_hparams[i][j]

    def estimatePopGradients(self):
        gradients = [[0] * self.num_hparm] * self.popsize
        for i in range(self.popsize):
            for j in range(self.num_hparm):
                gradients[i][j] = self.current_hparams[i][
                    j] - self.best_hparms_all_pop[j]
        return gradients

    def estimateLocalGradients(self):
        gradients = [[0] * self.num_hparm] * self.popsize
        for i in range(self.popsize):
            for j in range(self.num_hparm):
                gradients[i][j] = self.current_hparams[i][
                    j] - self.best_hparams_per_pop[i][j]
        return gradients

    def estimateMomemtum(self):
        popGrads = self.estimatePopGradients()
        localGrads = self.estimateLocalGradients()
        for i in range(self.popsize):
            for j in range(self.num_hparm):
                self.momentums[i][j] = (
                    1 - 3.0 * self.alpha / self.iteration
                ) * self.momentums[i][j] - self.alpha * localGrads[i][
                    j] - self.alpha * popGrads[i][j]

    def is_stop(self):
        return False

    def solutions(self):
        return self.current_hparams

    def feedback(self, params_list, reward_list):
        self._iteration = self._iteration + 1
        for i in range(self.popsize):
            if reward_list[i] < self.best_reward_per_pop[i]:
                self.best_hparams_per_pop[i] = copy.deepcopy(
                    self.current_hparams[i])
                self.best_reward_per_pop[i] = reward_list[i]
            if reward_list[i] < self.best_reward_all_pop:
                self.best_hparms_all_pop = self.current_hparams[i]
                self.best_reward_all_pop = reward_list[i]
        self.estimateMomemtum()
        for i in range(self.popsize):
            for j in range(len(self.init_input)):
                self.current_hparams[i][j] = self.current_hparams[i][
                    j] + self.alpha * self.momentums[i][j]
        self.smallPeturb()

    def optimal_solution(self):
        return self.best_hparms_all_pop

    def step(self, output_dir):
        solutions = self.solutions()

        params_cudas_dirs = []
        solution_results = []
        cnt = 0
        solutions_ckptdirs = {}
        mkdir(output_dir)
        for idx, solution in enumerate(solutions):
            cuda = self.is_cuda_free["free"][0]
            ckptdir = output_dir + "/ckpt-" + str(idx)
            log_file = output_dir + "/log-" + str(idx) + ".info"
            params_cudas_dirs.append([solution, cuda, ckptdir, log_file])
            solutions_ckptdirs[tuple(solution)] = ckptdir
            self.is_cuda_free["free"].remove(cuda)
            self.is_cuda_free["busy"].append(cuda)
            if len(params_cudas_dirs) == self.thread or cnt == int(
                    self.popsize / self.thread):
                tp = ThreadPool(len(params_cudas_dirs))
                solution_results += tp.map(self.evaluator.run,
                                           params_cudas_dirs)
                cnt += 1
                tp.close()
                tp.join()
                for param_cuda in params_cudas_dirs:
                    self.is_cuda_free["free"].append(param_cuda[1])
                    self.is_cuda_free["busy"].remove(param_cuda[1])
                params_cudas_dirs = []

        self.feedback(solutions, solution_results)

        return solutions_ckptdirs
