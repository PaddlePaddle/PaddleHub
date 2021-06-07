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

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from paddlehub.common.logger import logger

from videotag_tsn_lstm.resource.utils.config_utils import *
import videotag_tsn_lstm.resource.models as models
from videotag_tsn_lstm.resource.reader import get_reader
from videotag_tsn_lstm.resource.metrics import get_metrics
from videotag_tsn_lstm.resource.utils.utility import check_cuda
from videotag_tsn_lstm.resource.utils.utility import check_version


@moduleinfo(
    name="videotag_tsn_lstm",
    version="1.0.0",
    summary=
    "videotag_tsn_lstm is a video classification model, using TSN for feature extraction and AttentionLSTM for classification",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="video/classification",
)
class VideoTag(hub.Module):
    def _initialize(self):
        # add arg parser
        self.parser = argparse.ArgumentParser(
            description="Run the videotag_tsn_lstm module.",
            prog='hub run videotag_tsn_lstm',
            usage='%(prog)s',
            add_help=True)
        self.parser.add_argument('--use_gpu', type=ast.literal_eval, default=False, help='default use gpu.')
        self.parser.add_argument('--input_path', type=str, default=None, help='path of video data, single video')
        self._has_load = False

    def _extractor(self, args, exe, place):
        extractor_scope = fluid.Scope()
        with fluid.scope_guard(extractor_scope):
            extractor_startup_prog = fluid.Program()
            extractor_main_prog = fluid.Program()
            with fluid.program_guard(extractor_main_prog, extractor_startup_prog):
                extractor_config = parse_config(args.extractor_config)
                extractor_infer_config = merge_configs(extractor_config, 'infer', vars(args))

                # build model
                extractor_model = models.get_model("TSN", extractor_infer_config, mode='infer')
                extractor_model.build_input(use_dataloader=False)
                extractor_model.build_model()
                extractor_feeds = extractor_model.feeds()
                extractor_fetch_list = extractor_model.fetches()

                exe.run(extractor_startup_prog)

                logger.info('load extractor weights from {}'.format(args.extractor_weights))
                extractor_model.load_test_weights(exe, args.extractor_weights, extractor_main_prog)

                extractor_feeder = fluid.DataFeeder(place=place, feed_list=extractor_feeds)
        return extractor_main_prog, extractor_fetch_list, extractor_feeder, extractor_scope

    def _predictor(self, args, exe, place):
        predictor_scope = fluid.Scope()
        with fluid.scope_guard(predictor_scope):
            predictor_startup_prog = fluid.default_startup_program()
            predictor_main_prog = fluid.default_main_program()
            with fluid.program_guard(predictor_main_prog, predictor_startup_prog):
                # parse config
                predictor_config = parse_config(args.predictor_config)
                predictor_infer_config = merge_configs(predictor_config, 'infer', vars(args))

                predictor_model = models.get_model("AttentionLSTM", predictor_infer_config, mode='infer')
                predictor_model.build_input(use_dataloader=False)
                predictor_model.build_model()
                predictor_feeds = predictor_model.feeds()
                predictor_outputs = predictor_model.outputs()

                exe.run(predictor_startup_prog)

                logger.info('load lstm weights from {}'.format(args.predictor_weights))
                predictor_model.load_test_weights(exe, args.predictor_weights, predictor_main_prog)

                predictor_feeder = fluid.DataFeeder(place=place, feed_list=predictor_feeds)
                predictor_fetch_list = predictor_model.fetches()
        return predictor_main_prog, predictor_fetch_list, predictor_feeder, predictor_scope

    @runnable
    def run_cmd(self, argsv):
        args = self.parser.parse_args(argsv)
        results = self.classify(paths=[args.input_path], use_gpu=args.use_gpu)
        return results

    def classify(self, paths, use_gpu=False, threshold=0.5, top_k=10):
        """
        API of Classification.

        Args:
            paths (list[str]): the path of mp4s.
            use_gpu (bool): whether to use gpu or not.
            threshold (float): the result value >= threshold will be returned.
            top_k (int): the top k result will be returned.

        Returns:
            results (list[dict]): every dict includes the mp4 file path and prediction.
        """
        args = self.parser.parse_args([])
        # config the args in videotag_tsn_lstm
        args.use_gpu = use_gpu
        args.filelist = paths
        args.topk = top_k
        args.threshold = threshold
        args.extractor_config = os.path.join(self.directory, 'resource', 'configs', 'tsn.yaml')
        args.predictor_config = os.path.join(self.directory, 'resource', 'configs', 'attention_lstm.yaml')
        args.extractor_weights = os.path.join(self.directory, 'weights', 'tsn')
        args.predictor_weights = os.path.join(self.directory, 'weights', 'attention_lstm')
        args.label_file = os.path.join(self.directory, 'resource', 'label_3396.txt')

        check_cuda(args.use_gpu)
        check_version()

        if not self._has_load:
            self.place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            self.exe = fluid.Executor(self.place)
            self.extractor_main_prog, self.extractor_fetch_list, self.extractor_feeder, self.extractor_scope = self._extractor(
                args, self.exe, self.place)
            self.predictor_main_prog, self.predictor_fetch_list, self.predictor_feeder, self.predictor_scope = self._predictor(
                args, self.exe, self.place)
            self._has_load = True

        extractor_config = parse_config(args.extractor_config)
        extractor_infer_config = merge_configs(extractor_config, 'infer', vars(args))
        extractor_reader = get_reader("TSN", 'infer', extractor_infer_config)
        feature_list = []
        file_list = []

        for idx, data in enumerate(extractor_reader()):
            file_id = [item[-1] for item in data]
            feed_data = [item[:-1] for item in data]
            feature_out = self.exe.run(
                program=self.extractor_main_prog,
                fetch_list=self.extractor_fetch_list,
                feed=self.extractor_feeder.feed(feed_data),
                scope=self.extractor_scope)
            feature_list.append(feature_out)
            file_list.append(file_id)
            logger.info('========[Stage 1 Sample {} ] Tsn feature extractor finished======'.format(idx))

        # get AttentionLSTM input from Tsn output
        num_frames = 300
        predictor_feed_list = []
        for i in range(len(feature_list)):
            feature_out = feature_list[i]
            extractor_feature = feature_out[0]
            predictor_feed_data = [[extractor_feature[0].astype(float)[0:num_frames, :]]]
            predictor_feed_list.append((predictor_feed_data, file_list[i]))

        metrics_config = parse_config(args.predictor_config)
        metrics_config['MODEL']['topk'] = args.topk
        metrics_config['MODEL']['threshold'] = args.threshold
        predictor_metrics = get_metrics("AttentionLSTM".upper(), 'infer', metrics_config)
        predictor_metrics.reset()
        for idx, data in enumerate(predictor_feed_list):
            file_id = data[1]
            predictor_feed_data = data[0]
            final_outs = self.exe.run(
                program=self.predictor_main_prog,
                fetch_list=self.predictor_fetch_list,
                feed=self.predictor_feeder.feed(predictor_feed_data, ),
                scope=self.predictor_scope)
            logger.info('=======[Stage 2 Sample {} ] AttentionLSTM predict finished========'.format(idx))
            final_result_list = [item for item in final_outs] + [file_id]

            predictor_metrics.accumulate(final_result_list)
        results = predictor_metrics.finalize_and_log_out(label_file=args.label_file)
        return results


if __name__ == '__main__':
    test_module = VideoTag()
    print(test_module.run_cmd(argsv=['--input_path', "1.mp4", '--use_gpu', str(False)]))
