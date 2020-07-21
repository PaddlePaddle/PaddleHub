# -*- coding:utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
"""Finetuning on classification task """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
import paddlehub as hub


@moduleinfo(
    name="ernie_tiny_finetuned",
    version="1.0.0",
    summary="ERNIE tiny which was fine-tuned on the chnsenticorp dataset.",
    author="anonymous",
    author_email="",
    type="nlp/semantic_model")
class ERNIETinyFinetuned(hub.Module):
    def _initialize(self,
                    ckpt_dir="ckpt_chnsenticorp",
                    num_class=2,
                    max_seq_len=128,
                    use_gpu=False,
                    batch_size=1):
        self.ckpt_dir = os.path.join(self.directory, ckpt_dir)
        self.num_class = num_class
        self.MAX_SEQ_LEN = max_seq_len

        # Load Paddlehub ERNIE Tiny pretrained model
        self.module = hub.Module(name="ernie_tiny")
        inputs, outputs, program = self.module.context(
            trainable=True, max_seq_len=max_seq_len)

        self.vocab_path = self.module.get_vocab_path()

        # Download dataset and use accuracy as metrics
        # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
        # metric should be acc, f1 or matthews
        metrics_choices = ["acc"]

        # For ernie_tiny, it use sub-word to tokenize chinese sentence
        # If not ernie tiny, sp_model_path and word_dict_path should be set None
        reader = hub.reader.ClassifyReader(
            vocab_path=self.module.get_vocab_path(),
            max_seq_len=max_seq_len,
            sp_model_path=self.module.get_spm_path(),
            word_dict_path=self.module.get_word_dict_path())

        # Construct transfer learning network
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of module need
        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        # Setup runing config for PaddleHub Finetune API
        config = hub.RunConfig(
            use_data_parallel=False,
            use_cuda=use_gpu,
            batch_size=batch_size,
            checkpoint_dir=self.ckpt_dir,
            strategy=hub.AdamWeightDecayStrategy())

        # Define a classfication finetune task by PaddleHub's API
        self.cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=self.num_class,
            config=config,
            metrics_choices=metrics_choices)

    @serving
    def predict(self, data, return_result=False, accelerate_mode=True):
        """
        Get prediction results
        """
        run_states = self.cls_task.predict(
            data=data,
            return_result=return_result,
            accelerate_mode=accelerate_mode)
        results = [run_state.run_results for run_state in run_states]
        prediction = []
        for batch_result in results:
            # get predict index
            batch_result = np.argmax(batch_result, axis=2)[0]
            batch_result = batch_result.tolist()
            prediction += batch_result
        return prediction


if __name__ == "__main__":
    ernie_tiny = ERNIETinyFinetuned(
        ckpt_dir="../ckpt_chnsenticorp", num_class=2)

    # Data to be prdicted
    data = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
            ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]

    predictions = ernie_tiny.predict(data=data)
    for index, text in enumerate(data):
        print("%s\tpredict=%s" % (data[index][0], predictions[index]))
