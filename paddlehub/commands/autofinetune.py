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
from paddlehub.autofinetune.autoft import HAZero
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
            required=True,
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
            default="modelbased",
            help="Choices: fulltrail or modelbased.")
        self.arg_config_group.add_argument(
            "--tuning_strategy",
            type=str,
            default="pshe2",
            help="Choices: HAZero or PSHE2.")
        self.arg_config_group.add_argument(
            'opts',
            help='See utils/config.py for all options',
            default=None,
            nargs=argparse.REMAINDER)

    def convert_to_other_options(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command for finetuned task options config format error! Please check it: {}"
                .format(config_list))
        options_str = ""
        for key, value in zip(config_list[0::2], config_list[1::2]):
            options_str += "--" + key + "=" + value + " "
        return options_str

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
        self.arg_finetuned_task_group = self.parser.add_argument_group(
            title="Finetuned task config options",
            description=
            "Finetuned task configuration for controlling finetuned task behavior, not required"
        )

        self.add_params_file_arg()
        self.add_autoft_config_arg()

        if not argv[1:]:
            self.help()
            return False

        self.args = self.parser.parse_args(argv[1:])
        options_str = ""
        if self.args.opts is not None:
            options_str = self.convert_to_other_options(self.args.opts)

        if self.args.evaluate_choice.lower() == "fulltrail":
            evaluator = FullTrailEvaluator(
                self.args.param_file,
                self.fintunee_script,
                options_str=options_str)
        elif self.args.evaluate_choice.lower() == "modelbased":
            evaluator = ModelBasedEvaluator(
                self.args.param_file,
                self.fintunee_script,
                options_str=options_str)
        else:
            raise ValueError(
                "The evaluate %s is not defined!" % self.args.evaluate_choice)

        if self.args.tuning_strategy.lower() == "hazero":
            autoft = HAZero(
                evaluator,
                cudas=self.args.cuda,
                popsize=self.args.popsize,
                output_dir=self.args.output_dir)
        elif self.args.tuning_strategy.lower() == "pshe2":
            autoft = PSHE2(
                evaluator,
                cudas=self.args.cuda,
                popsize=self.args.popsize,
                output_dir=self.args.output_dir)
        else:
            raise ValueError("The tuning strategy %s is not defined!" %
                             self.args.tuning_strategy)

        run_round_cnt = 0
        solutions_modeldirs = {}
        print("PaddleHub Autofinetune starts.")
        while (not autoft.is_stop()) and run_round_cnt < self.args.round:
            print("PaddleHub Autofinetune starts round at %s." % run_round_cnt)
            output_dir = autoft._output_dir + "/round" + str(run_round_cnt)
            res = autoft.step(output_dir)
            solutions_modeldirs.update(res)
            evaluator.new_round()
            run_round_cnt = run_round_cnt + 1
        print("PaddleHub Autofinetune ends.")

        with open(autoft._output_dir + "/log_file.txt", "w") as f:
            best_hparams = evaluator.convert_params(autoft.get_best_hparams())
            print("The final best hyperparameters:")
            f.write("The final best hyperparameters:\n")
            for index, hparam_name in enumerate(autoft.hparams_name_list):
                print("%s=%s" % (hparam_name, best_hparams[index]))
                f.write(hparam_name + "\t:\t" + str(best_hparams[index]) + "\n")
            f.write("\n\n\n")
            f.write("\t".join(autoft.hparams_name_list) +
                    "\tsaved_params_dir\n\n")
            print(
                "The related infomation  about hyperparamemters searched are saved as %s/log_file.txt ."
                % autoft._output_dir)
            for solution, modeldir in solutions_modeldirs.items():
                param = evaluator.convert_params(solution)
                param = [str(p) for p in param]
                f.write("\t".join(param) + "\t" + modeldir + "\n\n")

        return True


command = AutoFineTuneCommand.instance()
