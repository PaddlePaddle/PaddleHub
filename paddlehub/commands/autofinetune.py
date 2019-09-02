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

import argparse
import io
import json
import os
import sys
import ast

import six
import pandas
import numpy as np

from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.common.arg_helper import add_argument, print_arguments
from paddlehub.autofinetune.autoft import PSHE2
from paddlehub.autofinetune.evaluator import FullTrailEvaluator
from paddlehub.autofinetune.evaluator import ModelBasedEvaluator
from paddlehub.common.logger import logger

import paddlehub as hub


class AutoFineTuneCommand(BaseCommand):
    name = "autofinetune"

    def __init__(self, name):
        super(AutoFineTuneCommand, self).__init__(name)
        self.show_in_help = True
        self.name = name
        self.description = "Paddlehub helps to finetune a task by searching hyperparameters automatically."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s <task to be fintuned in python script>' % (ENTRY,
                                                                   self.name),
            usage='%(prog)s',
            add_help=False)
        self.module = None

    def add_params_file_arg(self):
        self.arg_params_to_be_searched_group.add_argument(
            "--param_file",
            type=str,
            default=None,
            required=True,
            help=
            "Hyperparameters to be searched in the yaml format. The number of hyperparameters searched must be greater than 1."
        )

    def add_autoft_config_arg(self):
        self.arg_config_group.add_argument(
            "--popsize", type=int, default=5, help="Population size")
        self.arg_config_group.add_argument(
            "--cuda",
            type=ast.literal_eval,
            default=['0'],
            help="The list of gpu devices to be used")
        self.arg_config_group.add_argument(
            "--round", type=int, default=10, help="Number of searches")
        self.arg_config_group.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Directory to model checkpoint")
        self.arg_config_group.add_argument(
            "--evaluate_choice",
            type=str,
            default="fulltrail",
            help="Choices: fulltrail or modelbased.")

    def execute(self, argv):
        if not argv:
            print("ERROR: Please specify a script to be finetuned in python.\n")
            self.help()
            return False

        self.fintunee_script = argv[0]

        self.parser.prog = '%s %s %s' % (ENTRY, self.name, self.fintunee_script)
        self.arg_params_to_be_searched_group = self.parser.add_argument_group(
            title="Input options",
            description="Hyperparameters to be searched.")
        self.arg_config_group = self.parser.add_argument_group(
            title="Autofinetune config options",
            description=
            "Autofintune configuration for controlling autofinetune behavior, not required"
        )

        self.add_params_file_arg()
        self.add_autoft_config_arg()

        if not argv[1:]:
            self.help()
            return False

        self.args = self.parser.parse_args(argv[1:])
        if self.args.evaluate_choice.lower() == "fulltrail":
            evaluator = FullTrailEvaluator(self.args.param_file,
                                           self.fintunee_script)
        elif self.args.evaluate_choice.lower() == "modelbased":
            evaluator = ModelBasedEvaluator(self.args.param_file,
                                            self.fintunee_script)
        else:
            raise ValueError(
                "The evaluate %s is not defined!" % self.args.evaluate_choice)

        autoft = PSHE2(
            evaluator,
            cudas=self.args.cuda,
            popsize=self.args.popsize,
            output_dir=self.args.output_dir)

        run_round_cnt = 0
        solutions_ckptdirs = {}
        print("PaddleHub Autofinetune starts.")
        while (not autoft.is_stop()) and run_round_cnt < self.args.round:
            print("PaddleHub Autofinetune starts round at %s." % run_round_cnt)
            output_dir = autoft._output_dir + "/round" + str(run_round_cnt)
            res = autoft.step(output_dir)
            solutions_ckptdirs.update(res)
            evaluator.new_round()
            run_round_cnt = run_round_cnt + 1
        print("PaddleHub Autofinetune ends.")
        with open("./log_file.txt", "w") as f:
            best_choice = evaluator.convert_params(autoft.optimal_solution())
            print("The best hyperparameters:")
            f.write("The best hyperparameters:\n")
            param_name = []
            for idx, param in enumerate(evaluator.params["param_list"]):
                param_name.append(param["name"])
                f.write(param["name"] + "\t:\t" + str(best_choice[idx]) + "\n")
                print("%s : %s" % (param["name"], best_choice[idx]))
            f.write("\n\n\n")
            f.write("\t".join(param_name) + "\toutput_dir\n\n")

            logger.info(
                "The checkpont directory of programs ran with paramemters searched are saved as log_file.txt ."
            )
            print(
                "The checkpont directory of programs ran with paramemters searched are saved as log_file.txt ."
            )
            for solution, ckptdir in solutions_ckptdirs.items():
                param = evaluator.convert_params(solution)
                param = [str(p) for p in param]
                f.write("\t".join(param) + "\t" + ckptdir + "\n\n")

        return True


command = AutoFineTuneCommand.instance()
