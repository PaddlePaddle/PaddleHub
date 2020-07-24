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
import math

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

from ernie_gen_couplet.model.tokenizing_ernie import ErnieTokenizer
from ernie_gen_couplet.model.decode import beam_search_infilling
from ernie_gen_couplet.model.modeling_ernie_gen import ErnieModelForGeneration


@moduleinfo(
    name="ernie_gen",
    version="1.0.0",
    summary="",
    author="baidu-nlp",
    author_email="",
    type="nlp/text_generation",
)
class ErnieGen(hub.NLPPredictionModule):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        gen_checkpoint_path = os.path.join(self.directory, "assets",
                                           "ernie_gen_couplet")
        ernie_checkpint_path = os.path.join(self.directory, "assets", "ernie")
        ernie_cfg_path = os.path.join(ernie_checkpint_path, 'ernie_config.json')
        ernie_cfg = dict(json.loads(open(ernie_cfg_path).read()))
        ernie_vocab_path = os.path.join(ernie_checkpint_path, 'vocab.txt')
        ernie_vocab = {
            j.strip().split('\t')[0]: i
            for i, j in enumerate(open(ernie_vocab_path).readlines())
        }
        # self.tokenizer_path = os.path.join(self.directory, "assets", "tokenizer")

        with fluid.dygraph.guard(fluid.CPUPlace()):
            with fluid.unique_name.guard():
                # self.model = ErnieModelForGeneration.from_pretrained(self.ernie_checkpint_path)  # 重新加载预训练模型
                self.model = ErnieModelForGeneration(ernie_cfg)
                finetuned_states, _ = D.load_dygraph(gen_checkpoint_path)
                self.model.set_dict(finetuned_states)

        # self.tokenizer = ErnieTokenizer.from_pretrained(ernie_checkpint_path, mask_token=None)

        self.tokenizer = ErnieTokenizer(ernie_vocab)
        self.rev_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        self.rev_dict[self.tokenizer.pad_id] = ''  # replace [PAD]
        self.rev_dict[self.tokenizer.unk_id] = ''  # replace [PAD]
        self.rev_lookup = np.vectorize(lambda i: self.rev_dict[i])

        self.predict = self.generate

    def generate(self, texts, use_gpu=False, batch_size=1):
        """
        Get the right rolls from the left rolls.

        Args:
             texts(list): the left rolls.
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(list): the right rolls.
        """
        if use_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
            use_gpu = False
            logger.warning(
                "use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True"
            )
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()

        if texts and isinstance(texts, list):
            predicted_data = texts
        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")

        with fluid.dygraph.guard(place):
            self.model.eval()
            results = []
            # for text in predicted_data:
            batch_num = int(math.ceil(len(predicted_data) / batch_size))
            for i in range(batch_num):
                ids, sids = zip(*[
                    self.tokenizer.encode(text)
                    for text in predicted_data[i * batch_size:
                                               (i + 1) * batch_size]
                ])
                src_ids = D.to_variable(np.array(ids))
                src_sids = D.to_variable(np.array(sids))
                output_ids = beam_search_infilling(
                    self.model,
                    src_ids,
                    src_sids,
                    eos_id=self.tokenizer.sep_id,
                    sos_id=self.tokenizer.cls_id,
                    attn_id=self.tokenizer.vocab['[MASK]'],
                    max_decode_len=20,
                    max_encode_len=20,
                    beam_width=5,
                    tgt_type_id=1)
                output_str = self.rev_lookup(output_ids.numpy())

                for ostr in output_str.tolist():
                    if '[SEP]' in ostr:
                        ostr = ostr[:ostr.index('[SEP]')]
                    results.append(ostr)
        return results

    @serving
    def serving_method(self, texts, use_gpu=False):
        """
        Run as a service.
        """
        return self.generate(texts, use_gpu)


if __name__ == "__main__":
    module = ErnieGen()
    print(module.generate(['人增福寿年增岁']))
