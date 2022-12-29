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
import argparse
import contextlib
import os
import sys
from collections import namedtuple

import paddle
import paddle.nn as nn

import paddlehub as hub
from .model import Plato2InferModel
from .readers.nsp_reader import NSPReader
from .readers.plato_reader import PlatoReader
from .utils import gen_inputs
from .utils.args import parse_args
from .utils.args import str2bool
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving
from paddlehub.module.nlp_module import DataFormatError


@moduleinfo(
    name="plato2_en_large",
    version="1.1.0",
    summary=
    "A novel pre-training model for dialogue generation, incorporated with latent discrete variables for one-to-many relationship modeling.",
    author="baidu-nlp",
    author_email="",
    type="nlp/text_generation",
)
class Plato2(nn.Layer, hub.NLPPredictionModule):

    def __init__(self):
        """
        initialize with the necessary elements
        """
        super(Plato2, self).__init__()
        args = self.setup_args()

        if args.num_layers == 24:
            n_head = 16
            hidden_size = 1024
        elif args.num_layers == 32:
            n_head = 32
            hidden_size = 2048
        else:
            raise ValueError('The pre-trained model only support 24 or 32 layers, '
                             'but received num_layers=%d.' % args.num_layers)

        self.plato_reader = PlatoReader(args)
        nsp_reader = NSPReader(args)
        self.model = Plato2InferModel(nsp_reader, args.num_layers, n_head, hidden_size)
        state_dict = paddle.load(args.init_from_ckpt)
        self.model.set_state_dict(state_dict)
        self.model.eval()
        self.Example = namedtuple("Example", ["src", "data_id"])
        self.latent_type_size = args.latent_type_size
        self._interactive_mode = False

    def setup_args(self):
        """
        Setup arguments.
        """
        ckpt_path = os.path.join(self.directory, 'assets', '32L.pdparams')
        vocab_path = os.path.join(self.directory, 'assets', 'vocab.txt')
        spm_model_file = os.path.join(self.directory, 'assets', 'spm.model')

        # ArgumentParser.parse_args use argv[1:], it will drop the first one arg, so the first one in sys.argv should be ""
        sys.argv = [
            "--empty",
            "--spm_model_file",
            "%s" % spm_model_file,
            "--vocab_path",
            "%s" % vocab_path,
        ]

        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("Model")
        group.add_argument("--init_from_ckpt", type=str, default=ckpt_path)
        group.add_argument("--vocab_size", type=int, default=8001)
        group.add_argument("--latent_type_size", type=int, default=20)
        group.add_argument("--num_layers", type=int, default=32)

        group = parser.add_argument_group("Task")
        group.add_argument("--is_cn", type=str2bool, default=False)

        NSPReader.add_cmdline_args(parser)

        args = parse_args(parser)
        args.batch_size *= args.latent_type_size

        return args

    @serving
    @paddle.no_grad()
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
            example = self.Example(src=text.replace("\t", " [SEP] "), data_id=0)
            record = self.plato_reader._convert_example_to_record(example, is_infer=True)
            data = self.plato_reader._pad_batch_records([record], is_infer=True)
            inputs = gen_inputs(data, self.latent_type_size)
            inputs['tgt_ids'] = inputs['tgt_ids'].astype('int64')
            pred = self.model(inputs)[0]  # batch_size is 1
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
        self.parser = argparse.ArgumentParser(description='Run the %s module.' % self.name,
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
