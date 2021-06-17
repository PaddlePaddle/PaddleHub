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
import json
import argparse
import os

import numpy as np
import paddle
import paddlehub as hub
from paddlehub.module.module import runnable
from paddlehub.module.nlp_module import DataFormatError
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
from paddlenlp.transformers import ErnieTokenizer, ErnieForGeneration

from ernie_gen_poetry.decode import beam_search_infilling


@moduleinfo(
    name="ernie_gen_poetry",
    version="1.1.0",
    summary=
    "ERNIE-GEN is a multi-flow language generation framework for both pre-training and fine-tuning. This module has fine-tuned for poetry generation task.",
    author="baidu-nlp",
    author_email="",
    type="nlp/text_generation",
)
class ErnieGen(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        assets_path = os.path.join(self.directory, "assets")
        gen_checkpoint_path = os.path.join(assets_path, "ernie_gen_poetry.pdparams")
        self.model = ErnieForGeneration.from_pretrained("ernie-1.0")
        model_state = paddle.load(gen_checkpoint_path)
        self.model.set_dict(model_state)
        self.tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
        self.rev_dict = self.tokenizer.vocab.idx_to_token
        self.rev_dict[self.tokenizer.vocab['[PAD]']] = ''  # replace [PAD]
        self.rev_dict[self.tokenizer.vocab['[UNK]']] = ''  # replace [PAD]
        self.rev_lookup = np.vectorize(lambda i: self.rev_dict[i])

    @serving
    def generate(self, texts, use_gpu=False, beam_width=5):
        """
        Get the continuation of the input poetry.

        Args:
             texts(list): the front part of a poetry.
             use_gpu(bool): whether use gpu to predict or not
             beam_width(int): the beam search width.

        Returns:
             results(list): the poetry continuations.
        """
        if texts and isinstance(texts, list) and all(texts) and all([isinstance(text, str) for text in texts]):
            predicted_data = texts
        else:
            raise ValueError("The input texts should be a list with nonempty string elements.")
        for i, text in enumerate(texts):
            if '，' not in text or '。' not in text:
                logger.warning(
                    "The input text: %s, does not contain '，' or '。', which is not a complete verse and may result in magic output"
                    % text)
            else:
                front, rear = text[:-1].split('，')
                if len(front) != len(rear):
                    logger.warning(
                        "The input text: %s, is no antithetical parallelism, which may result in magic output" % text)

            for char in text:
                if not '\u4e00' <= char <= '\u9fff' and char not in ['，', '。']:
                    logger.warning(
                        "The input text: %s, contains characters not Chinese or ‘，’ '。', which may result in magic output"
                        % text)
                break

        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            use_gpu = False
            logger.warning(
                "use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True"
            )
        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')
        self.model.eval()
        results = []
        for text in predicted_data:
            sample_results = []
            encode_text = self.tokenizer.encode(text)
            src_ids = paddle.to_tensor(encode_text['input_ids']).unsqueeze(0)
            src_sids = paddle.to_tensor(encode_text['token_type_ids']).unsqueeze(0)
            output_ids = beam_search_infilling(self.model,
                                               src_ids,
                                               src_sids,
                                               eos_id=self.tokenizer.vocab['[SEP]'],
                                               sos_id=self.tokenizer.vocab['[CLS]'],
                                               attn_id=self.tokenizer.vocab['[MASK]'],
                                               pad_id=self.tokenizer.vocab['[PAD]'],
                                               unk_id=self.tokenizer.vocab['[UNK]'],
                                               vocab_size=len(self.tokenizer.vocab),
                                               max_decode_len=80,
                                               max_encode_len=20,
                                               beam_width=beam_width,
                                               tgt_type_id=1)
            output_str = self.rev_lookup(output_ids[0])

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
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU for prediction")

        self.arg_config_group.add_argument('--beam_width', type=int, default=5, help="the beam search width")

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


if __name__ == "__main__":
    module = ErnieGen()
    for result in module.generate(['昔年旅南服，始识王荆州。', '高名出汉阴，禅阁跨香岑。'], beam_width=5, use_gpu=True):
        print(result)
