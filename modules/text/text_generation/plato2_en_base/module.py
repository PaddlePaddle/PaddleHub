# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import ast
import os
import json
import sys
import argparse
import contextlib
from collections import namedtuple

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import runnable
from paddlehub.module.nlp_module import DataFormatError
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving

import plato2_en_base.models as plato_models
from plato2_en_base.tasks.dialog_generation import DialogGeneration
from plato2_en_base.utils import check_cuda, Timer
from plato2_en_base.utils.args import parse_args


@moduleinfo(
    name="plato2_en_base",
    version="1.0.0",
    summary=
    "A novel pre-training model for dialogue generation, incorporated with latent discrete variables for one-to-many relationship modeling.",
    author="baidu-nlp",
    author_email="",
    type="nlp/text_generation",
)
class Plato(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            raise RuntimeError("The module only support GPU. Please set the environment variable CUDA_VISIBLE_DEVICES.")

        args = self.setup_args()
        self.task = DialogGeneration(args)
        self.model = plato_models.create_model(args, fluid.CUDAPlace(0))
        self.Example = namedtuple("Example", ["src", "data_id"])
        self._interactive_mode = False

    def setup_args(self):
        """
        Setup arguments.
        """
        assets_path = os.path.join(self.directory, "assets")
        vocab_path = os.path.join(assets_path, "vocab.txt")
        init_pretraining_params = os.path.join(assets_path, "24L", "Plato")
        spm_model_file = os.path.join(assets_path, "spm.model")
        nsp_inference_model_path = os.path.join(assets_path, "24L", "NSP")
        config_path = os.path.join(assets_path, "24L.json")

        # ArgumentParser.parse_args use argv[1:], it will drop the first one arg, so the first one in sys.argv should be ""
        sys.argv = [
            "", "--model", "Plato", "--vocab_path",
            "%s" % vocab_path, "--do_lower_case", "False", "--init_pretraining_params",
            "%s" % init_pretraining_params, "--spm_model_file",
            "%s" % spm_model_file, "--nsp_inference_model_path",
            "%s" % nsp_inference_model_path, "--ranking_score", "nsp_score", "--do_generation", "True", "--batch_size",
            "1", "--config_path",
            "%s" % config_path
        ]

        parser = argparse.ArgumentParser()
        plato_models.add_cmdline_args(parser)
        DialogGeneration.add_cmdline_args(parser)
        args = parse_args(parser)

        args.load(args.config_path, "Model")
        args.run_infer = True  # only build infer program

        return args

    @serving
    def generate(self, texts):
        """
        Get the robot responses of the input texts.

        Args:
             texts(list or str): If not in the interactive mode, texts should be a list in which every element is the chat context separated with '\t'.
                                 Otherwise, texts shoule be one sentence. The module can get the context automatically.

        Returns:
             results(list): the robot responses.
        """
        if not texts:
            return []
        if self._interactive_mode:
            if isinstance(texts, str):
                self.context.append(texts.strip())
                texts = [" [SEP] ".join(self.context[-self.max_turn:])]
            else:
                raise ValueError("In the interactive mode, the input data should be a string.")
        elif not isinstance(texts, list):
            raise ValueError("If not in the interactive mode, the input data should be a list.")

        bot_responses = []
        for i, text in enumerate(texts):
            example = self.Example(src=text.replace("\t", " [SEP] "), data_id=i)
            record = self.task.reader._convert_example_to_record(example, is_infer=True)
            data = self.task.reader._pad_batch_records([record], is_infer=True)
            pred = self.task.infer_step(self.model, data)[0]  # batch_size is 1
            bot_response = pred["response"]  # ignore data_id and score
            bot_responses.append(bot_response)

        if self._interactive_mode:
            self.context.append(bot_responses[0].strip())
        return bot_responses

    @contextlib.contextmanager
    def interactive_mode(self, max_turn=6):
        """
        Enter the interactive mode.

        Args:
            max_turn(int): the max dialogue turns. max_turn = 1 means the robot can only remember the last one utterance you have said.
        """
        self._interactive_mode = True
        self.max_turn = max_turn
        self.context = []
        yield
        self.context = []
        self._interactive_mode = False

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description='Run the %s module.' % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, optional.")

        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.generate(texts=input_data)

        return results


if __name__ == "__main__":
    module = Plato()
    for result in module.generate(["Hello", "Hello\thi, nice to meet you, my name is tom\tso your name is tom?"]):
        print(result)
    with module.interactive_mode(max_turn=3):
        while True:
            human_utterance = input()
            robot_utterance = module.generate(human_utterance)
            print("Robot: %s" % robot_utterance[0])
