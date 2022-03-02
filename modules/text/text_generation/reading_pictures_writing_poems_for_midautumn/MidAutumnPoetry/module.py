# coding:utf-8
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
import json

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import runnable
from paddlehub.module.nlp_module import DataFormatError
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving

import argparse
import os
import numpy as np

import paddle.fluid.dygraph as D

from reading_pictures_writing_poems_for_midautumn.MidAutumnPoetry.model.tokenizing_ernie import ErnieTokenizer
from reading_pictures_writing_poems_for_midautumn.MidAutumnPoetry.model.decode import beam_search_infilling
from reading_pictures_writing_poems_for_midautumn.MidAutumnPoetry.model.modeling_ernie_gen import ErnieModelForGeneration


@moduleinfo(
    name="MidAutumnPoetry",
    version="1.0.0",
    summary="",
    author="郑博培，彭兆帅",
    author_email="2733821739@qq.com，1084667371@qq.com",
    type="nlp/text_generation",
)
class ErnieGen(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        assets_path = os.path.join(self.directory, "assets")
        gen_checkpoint_path = os.path.join(assets_path, "ernie_gen")
        ernie_cfg_path = os.path.join(assets_path, 'ernie_config.json')
        with open(ernie_cfg_path, encoding='utf8') as ernie_cfg_file:
            ernie_cfg = dict(json.loads(ernie_cfg_file.read()))
        ernie_vocab_path = os.path.join(assets_path, 'vocab.txt')
        with open(ernie_vocab_path, encoding='utf8') as ernie_vocab_file:
            ernie_vocab = {j.strip().split('\t')[0]: i for i, j in enumerate(ernie_vocab_file.readlines())}

        with fluid.dygraph.guard(fluid.CPUPlace()):
            with fluid.unique_name.guard():
                self.model = ErnieModelForGeneration(ernie_cfg)
                finetuned_states, _ = D.load_dygraph(gen_checkpoint_path)
                self.model.set_dict(finetuned_states)

        self.tokenizer = ErnieTokenizer(ernie_vocab)
        self.rev_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        self.rev_dict[self.tokenizer.pad_id] = ''  # replace [PAD]
        self.rev_dict[self.tokenizer.unk_id] = ''  # replace [PAD]
        self.rev_lookup = np.vectorize(lambda i: self.rev_dict[i])

    @serving
    def generate(self, texts, use_gpu=False, beam_width=5):
        """
        Get the predict result from the input texts.

        Args:
             texts(list): the input texts.
             use_gpu(bool): whether use gpu to predict or not
             beam_width(int): the beam search width.

        Returns:
             results(list): the predict result.
        """
        if texts and isinstance(texts, list) and all(texts) and all([isinstance(text, str) for text in texts]):
            predicted_data = texts
        else:
            raise ValueError("The input texts should be a list with nonempty string elements.")

        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            use_gpu = False
            logger.warning(
                "use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True"
            )
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        with fluid.dygraph.guard(place):
            self.model.eval()
            results = []
            for text in predicted_data:
                sample_results = []
                ids, sids = self.tokenizer.encode(text)
                src_ids = D.to_variable(np.expand_dims(ids, 0))
                src_sids = D.to_variable(np.expand_dims(sids, 0))
                output_ids = beam_search_infilling(
                    self.model,
                    src_ids,
                    src_sids,
                    eos_id=self.tokenizer.sep_id,
                    sos_id=self.tokenizer.cls_id,
                    attn_id=self.tokenizer.vocab['[MASK]'],
                    max_decode_len=50,
                    max_encode_len=50,
                    beam_width=beam_width,
                    tgt_type_id=1)
                output_str = self.rev_lookup(output_ids[0].numpy())

                for ostr in output_str.tolist():
                    if '[SEP]' in ostr:
                        ostr = ostr[:ostr.index('[SEP]')]
                    sample_results.append("".join(ostr))
                results.append(sample_results)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU for prediction")

        self.arg_config_group.add_argument('--beam_width', type=int, default=5, help="the beam search width")

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

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.generate(texts=input_data, use_gpu=args.use_gpu, beam_width=args.beam_width)

        return results
