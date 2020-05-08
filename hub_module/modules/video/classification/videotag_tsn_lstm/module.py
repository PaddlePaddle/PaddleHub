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
import ast
import os

import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from videotag_tsn_lstm.resource.predict import predict


@moduleinfo(
    name="videotag_tsn_lstm",
    version="1.0.0",
    summary=
    "videoTag_TSN_LSTM is a video classification model, using TSN for feature extraction and AttentionLSTM for classification",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="video/classification",
)
class videoTag(hub.Module):
    def _initialize(self):
        # add arg parser
        self.parser = argparse.ArgumentParser(
            description="Run the videoTag_TSN_LSTM module.",
            prog='hub run videoTag_TSN_LSTM',
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help='default use gpu.')
        self.parser.add_argument(
            '--video_path',
            type=str,
            default=None,
            help='path of video data, single video')
        self.parser.add_argument(
            '--filelist',
            type=str,
            default=None,
            help='path of video data, multiple video')
        self.parser.add_argument(
            '--save_dir', type=str, default=None, help='output file path')

    @runnable
    def run_cmd(self, argsv):
        args = self.parser.parse_args(argsv)
        if args.filelist:
            args.single_file = False
        elif args.video_path:
            args.single_file = True
        else:
            raise ValueError(
                "Neither filelist and video_path has been specifed.")
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

    def classification(self,
                       video_path=None,
                       filelist=None,
                       use_gpu=False,
                       save_dir=None):
        argsv = [
            '--video_path', video_path, '--filelist', filelist, '--use_gpu',
            str(use_gpu), '--save_dir', save_dir
        ]
        return self.run_cmd(argsv)


if __name__ == '__main__':
    test_module = videoTag()
    print(test_module.run_cmd())
