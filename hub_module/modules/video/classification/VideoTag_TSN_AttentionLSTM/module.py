# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import ast
import os

import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable

from VideoTag_TSN_AttentionLSTM.resource.predict import predict


@moduleinfo(
    name="VideoTag_TSN_AttentionLSTM",
    version="1.0.0",
    summary=
    "VideoTag_TSN_AttentionLSTM is a video classification model, using TSN for feature extraction and AttentionLSTM for classification",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="video/classification",
)
class videoTag(hub.Module):
    def _initialize(self):
        # add arg parser
        self.parser = argparse.ArgumentParser(
            description="Run the VideoTag_TSN_AttentionLSTM module.",
            prog='hub run VideoTag_TSN_AttentionLSTM',
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help='default use gpu.')
        self.parser.add_argument(
            '--input_path',
            type=str,
            default=None,
            help='path of video data, single video')

    @runnable
    def run_cmd(self, argsv):
        args = self.parser.parse_args(argsv)
        results = self.classification(
            paths=[args.input_path], use_gpu=args.use_gpu)
        return results

    def classification(self, paths, use_gpu=False, top_k=10, save_dir=None):
        args = self.parser.parse_args([])
        # config the args in VideoTag_TSN_AttentionLSTM
        args.use_gpu = use_gpu
        args.save_dir = save_dir
        args.filelist = paths
        args.topk = top_k
        args.extractor_config = os.path.join(self.directory, 'resource',
                                             'configs', 'tsn.yaml')
        args.predictor_config = os.path.join(self.directory, 'resource',
                                             'configs', 'attention_lstm.yaml')
        args.extractor_weights = os.path.join(self.directory, 'weights', 'tsn')
        args.predictor_weights = os.path.join(self.directory, 'weights',
                                              'attention_lstm')
        args.label_file = os.path.join(self.directory, 'resource',
                                       'label_3396.txt')
        results = predict(args)
        return results


if __name__ == '__main__':
    test_module = videoTag()
    print(
        test_module.run_cmd(
            argsv=['--input_path', "1.mp4", '--use_gpu',
                   str(False)]))
