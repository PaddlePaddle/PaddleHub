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
import os
import random
import yaml

from paddlehub.common.logger import logger

REWARD_SUM = 10000


class AutoFineTuneEvaluator(object):
    def __init__(self, params_file, finetunee_script):
        with io.open(params_file, 'r', encoding='utf8') as f:
            self.params = yaml.safe_load(f)
        self.finetunee_script = finetunee_script

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
        params = args[0][0]
        num_cuda = args[0][1]
        ckpt_dir = args[0][2]
        log_file = args[0][3]
        params = self.convert_params(params)
        if not self.is_valid_params(params):
            return REWARD_SUM

        param_str = self.format_params_str(params)
        os.system("touch " + log_file)
        run_cmd = "export FLAGS_eager_delete_tensor_gb=0.0; export CUDA_VISIBLE_DEVICES=%s; python -u %s --checkpoint_dir=%s %s >%s 2>&1" % \
                    (num_cuda, self.finetunee_script, ckpt_dir, param_str, log_file)

        try:
            f = os.popen(run_cmd)
            eval_result = float(f.readlines()[-1])
        except:
            logger.warning(
                "Program which was ran with hyperparameters as %s was crashed!"
                % param_str.replace("--", ""))
            eval_result = 0.0
        return self.get_reward(eval_result)
