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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import hashlib
import math
import os
import random
import six
import yaml

from paddlehub.common.dir import HUB_HOME
from paddlehub.common.logger import logger
from paddlehub.common.utils import is_windows, mkdir

REWARD_SUM = 1
TMP_HOME = os.path.join(HUB_HOME, "tmp")

if six.PY3:
    INF = math.inf
else:
    INF = float("inf")


def report_final_result(result):
    trial_id = os.environ.get("PaddleHub_AutoDL_Trial_ID")
    # tmp.txt is to record the eval results for trials
    mkdir(TMP_HOME)
    tmp_file = os.path.join(TMP_HOME, "tmp.txt")
    with open(tmp_file, 'a') as file:
        file.write(trial_id + "\t" + str(float(result)) + "\n")


def unique_name():
    seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+=-"
    x = []
    for idx in range(4):
        x.append(random.choice(seed))
    rand_str = "".join(x)
    return rand_str


class BaseEvaluator(object):
    def __init__(self, params_file, finetunee_script, options_str=""):
        with io.open(params_file, 'r', encoding='utf8') as f:
            self.params = yaml.safe_load(f)
        self.finetunee_script = finetunee_script
        self.model_rewards = {}
        self.options_str = options_str

    def get_init_params(self):
        init_params = []
        for param in self.params["param_list"]:
            init_params.append(param['init_value'])
        init_params = self.inverse_convert_params(init_params)
        return init_params

    def get_reward(self, result_output):
        return REWARD_SUM - float(result_output)

    def is_valid_params(self, params):
        for i in range(0, len(self.params["param_list"])):
            if params[i] < float(self.params["param_list"][i]["greater_than"]):
                return False
            if params[i] > float(self.params["param_list"][i]["lower_than"]):
                return False
        return True

    def convert_params(self, params):
        cparams = []
        for i in range(0, len(self.params["param_list"])):
            cparams.append(
                float(self.params["param_list"][i]["greater_than"] +
                      (params[i] + 1.0) / 2.0 *
                      (self.params["param_list"][i]["lower_than"] -
                       self.params["param_list"][i]["greater_than"])))
            if cparams[i] <= float(
                    self.params["param_list"][i]["greater_than"]):
                cparams[i] = float(self.params["param_list"][i]["greater_than"])
            if cparams[i] >= float(self.params["param_list"][i]["lower_than"]):
                cparams[i] = float(self.params["param_list"][i]["lower_than"])
            if self.params["param_list"][i]["type"] == "int":
                cparams[i] = int(cparams[i])
        return cparams

    def inverse_convert_params(self, params):
        cparams = []
        for i in range(0, len(self.params["param_list"])):
            cparams.append(
                float(
                    -1.0 + 2.0 *
                    (params[i] - self.params["param_list"][i]["greater_than"]) /
                    (self.params["param_list"][i]["lower_than"] -
                     self.params["param_list"][i]["greater_than"])))
            if cparams[i] <= -1.0:
                cparams[i] = -1.0
            if cparams[i] >= 1.0:
                cparams[i] = 1.0
        return cparams

    def format_params_str(self, params):
        param_str = "--%s=%s" % (self.params["param_list"][0]["name"],
                                 params[0])
        for i in range(1, len(self.params["param_list"])):
            param_str = "%s --%s=%s" % (
                param_str, self.params["param_list"][i]["name"], str(params[i]))
        return param_str

    def run(self, *args):
        raise NotImplementedError

    def new_round(self):
        pass


class FullTrailEvaluator(BaseEvaluator):
    def __init__(self, params_file, finetunee_script, options_str=""):
        super(FullTrailEvaluator, self).__init__(
            params_file, finetunee_script, options_str=options_str)

    def run(self, *args):
        params = args[0][0]
        num_cuda = args[0][1]
        saved_params_dir = args[0][2]
        log_file = args[0][3]
        params = self.convert_params(params)
        if not self.is_valid_params(params):
            return REWARD_SUM

        param_str = self.format_params_str(params)
        f = open(log_file, "w")
        f.close()

        if is_windows():
            run_cmd = "set FLAGS_eager_delete_tensor_gb=0.0&set CUDA_VISIBLE_DEVICES=%s&python -u %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                    (num_cuda, self.finetunee_script, saved_params_dir, param_str, self.options_str, log_file)
        else:
            run_cmd = "export FLAGS_eager_delete_tensor_gb=0.0; export CUDA_VISIBLE_DEVICES=%s; python -u %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                    (num_cuda, self.finetunee_script, saved_params_dir, param_str, self.options_str, log_file)

        try:
            #  set temp environment variable to record the eval results for trials
            rand_str = unique_name()
            os.environ['PaddleHub_AutoDL_Trial_ID'] = rand_str

            os.system(run_cmd)

            eval_result = []
            tmp_file = os.path.join(TMP_HOME, 'tmp.txt')
            with open(tmp_file, 'r') as file:
                for line in file:
                    data = line.strip().split("\t")
                    if rand_str == data[0]:
                        eval_result = float(data[1])
            if eval_result == []:
                print(
                    "WARNING: Program which was ran with hyperparameters as %s was crashed!"
                    % param_str.replace("--", ""))
                eval_result = 0.0
        except:
            print(
                "WARNING: Program which was ran with hyperparameters as %s was crashed!"
                % param_str.replace("--", ""))
            eval_result = 0.0

        reward = self.get_reward(eval_result)
        self.model_rewards[saved_params_dir] = reward
        return reward


class PopulationBasedEvaluator(BaseEvaluator):
    def __init__(self, params_file, finetunee_script, options_str=""):
        super(PopulationBasedEvaluator, self).__init__(
            params_file, finetunee_script, options_str=options_str)
        self.half_best_model_path = []
        self.run_count = 0

    def run(self, *args):
        params = args[0][0]
        num_cuda = args[0][1]
        saved_params_dir = args[0][2]
        log_file = args[0][3]
        params = self.convert_params(params)
        if not self.is_valid_params(params):
            return REWARD_SUM

        param_str = self.format_params_str(params)
        f = open(log_file, "w")
        f.close()

        if len(self.half_best_model_path) > 0:
            model_path = self.half_best_model_path[self.run_count % len(
                self.half_best_model_path)]
            if is_windows():
                run_cmd = "set FLAGS_eager_delete_tensor_gb=0.0&set CUDA_VISIBLE_DEVICES=%s&python -u %s --epochs=1 --model_path %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                        (num_cuda, self.finetunee_script, model_path, saved_params_dir, param_str, self.options_str, log_file)
            else:
                run_cmd = "export FLAGS_eager_delete_tensor_gb=0.0; export CUDA_VISIBLE_DEVICES=%s; python -u %s --epochs=1 --model_path %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                        (num_cuda, self.finetunee_script, model_path, saved_params_dir, param_str, self.options_str, log_file)

        else:
            if is_windows():
                run_cmd = "set FLAGS_eager_delete_tensor_gb=0.0&set CUDA_VISIBLE_DEVICES=%s&python -u %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                        (num_cuda, self.finetunee_script, saved_params_dir, param_str, self.options_str, log_file)
            else:
                run_cmd = "export FLAGS_eager_delete_tensor_gb=0.0; export CUDA_VISIBLE_DEVICES=%s; python -u %s --saved_params_dir=%s %s %s >%s 2>&1" % \
                        (num_cuda, self.finetunee_script, saved_params_dir, param_str, self.options_str, log_file)

        self.run_count += 1

        try:
            #  set temp environment variable to record the eval results for trials
            rand_str = unique_name()
            os.environ['PaddleHub_AutoDL_Trial_ID'] = rand_str

            os.system(run_cmd)

            eval_result = []
            tmp_file = os.join.path(TMP_HOME, 'tmp.txt')
            with open(tmp_file, 'r') as file:
                for line in file:
                    data = line.strip().split("\t")
                    if rand_str == data[0]:
                        eval_result = float(data[1])
            if eval_result == []:
                print(
                    "WARNING: Program which was ran with hyperparameters as %s was crashed!"
                    % param_str.replace("--", ""))
                eval_result = 0.0
        except:
            print(
                "WARNING: Program which was ran with hyperparameters as %s was crashed!"
                % param_str.replace("--", ""))
            eval_result = 0.0

        reward = self.get_reward(eval_result)
        self.model_rewards[saved_params_dir] = reward
        return reward

    def new_round(self):
        """update half_best_model"""
        half_size = int(len(self.model_rewards) / 2)
        if half_size < 1:
            half_size = 1
        self.half_best_model_path = list({
            key
            for key in sorted(
                self.model_rewards, key=self.model_rewards.get, reverse=False)
            [:half_size]
        })
        self.model_rewards = {}
