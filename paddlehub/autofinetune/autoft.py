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
import cma
import json
import numpy
import os
import time

from paddlehub.common.logger import logger
from paddlehub.common.utils import mkdir


class AutoFineTune(object):
    def __init__(self,
                 evaluator,
                 cudas=["0"],
                 popsize=1,
                 output_dir=None,
                 sigma=0.2):

        self._num_thread = len(cudas)
        self._popsize = popsize
        self._sigma = sigma
        self.cudas = cudas
        self.is_cuda_free = {"free": [], "busy": []}
        self.is_cuda_free["free"] = cudas

        self.evaluator = evaluator
        init_input = evaluator.get_init_params()
        self.evolution_stratefy = cma.CMAEvolutionStrategy(
            init_input, sigma, {
                'popsize': self.popsize,
                'bounds': [-1, 1],
                'AdaptSigma': True,
                'verb_disp': 1,
                'verb_time': 'True',
            })
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
    def sigma(self):
        return self._sigma

    @property
    def output_dir(self):
        self._output_dir

    def set_output_dir(self, output_dir=None):
        if output_dir is not None:
            output_dir = output_dir
        else:
            output_dir = self._output_dir
        return output_dir

    def is_stop(self):
        return self.evolution_stratefy.stop()

    def solutions(self):
        return self.evolution_stratefy.ask()

    def feedback(self, params_list, reward_list):
        self.evolution_stratefy.tell(params_list, reward_list)
        self.evolution_stratefy.disp()

    def optimal_solution(self):
        return list(self.evolution_stratefy.result.xbest)

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
