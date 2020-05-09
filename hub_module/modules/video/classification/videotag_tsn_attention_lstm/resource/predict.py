#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
import paddle.fluid as fluid

from videotag_tsn_attention_lstm.resource.utils.config_utils import *
import videotag_tsn_attention_lstm.resource.models as models
from videotag_tsn_attention_lstm.resource.reader import get_reader
from videotag_tsn_attention_lstm.resource.metrics import get_metrics
from videotag_tsn_attention_lstm.resource.utils.utility import check_cuda
from videotag_tsn_attention_lstm.resource.utils.utility import check_version

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def predict(args):
    """
    Video classification of 3k Chinese tags.
    videotag_tsn_attention_lstm (named as VideoTag_featureModel_predictModel)
    two stages in our model:
        1. extract feature from input video(mp4 format) using TSN model
        2. predict classification results from extracted feature  using AttentionLSTM model
    we implement this using two name scopes, ie. extractor_scope and predictor_scope.
    """
    check_cuda(args.use_gpu)
    check_version()
    extractor_scope = fluid.Scope()
    with fluid.scope_guard(extractor_scope):
        extractor_startup_prog = fluid.Program()
        extractor_main_prog = fluid.Program()
        with fluid.program_guard(extractor_main_prog, extractor_startup_prog):
            extractor_config = parse_config(args.extractor_config)
            extractor_infer_config = merge_configs(extractor_config, 'infer',
                                                   vars(args))

            # build model
            extractor_model = models.get_model(
                "TSN", extractor_infer_config, mode='infer')
            extractor_model.build_input(use_dataloader=False)
            extractor_model.build_model()
            extractor_feeds = extractor_model.feeds()
            extractor_fetch_list = extractor_model.fetches()

            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)

            exe.run(extractor_startup_prog)

            logger.info('load extractor weights from {}'.format(
                args.extractor_weights))
            extractor_model.load_test_weights(exe, args.extractor_weights,
                                              extractor_main_prog)

            # get reader and metrics
            extractor_reader = get_reader("TSN", 'infer',
                                          extractor_infer_config)
            extractor_feeder = fluid.DataFeeder(
                place=place, feed_list=extractor_feeds)

            feature_list = []
            file_list = []
            for idx, data in enumerate(extractor_reader()):
                file_id = [item[-1] for item in data]
                feed_data = [item[:-1] for item in data]
                feature_out = exe.run(
                    fetch_list=extractor_fetch_list,
                    feed=extractor_feeder.feed(feed_data))
                feature_list.append(feature_out)
                file_list.append(file_id)
                logger.info(
                    '========[Stage 1 Sample {} ] Tsn feature extractor finished======'
                    .format(idx))

    # get AttentionLSTM input from Tsn output
    num_frames = 300
    predictor_feed_list = []
    for i in range(len(feature_list)):
        feature_out = feature_list[i]
        extractor_feature = feature_out[0]
        predictor_feed_data = [[
            extractor_feature[0].astype(float)[0:num_frames, :]
        ]]
        predictor_feed_list.append((predictor_feed_data, file_list[i]))

    predictor_scope = fluid.Scope()
    with fluid.scope_guard(predictor_scope):
        predictor_startup_prog = fluid.default_startup_program()
        predictor_main_prog = fluid.default_main_program()
        with fluid.program_guard(predictor_main_prog, predictor_startup_prog):
            # parse config
            predictor_config = parse_config(args.predictor_config)
            predictor_infer_config = merge_configs(predictor_config, 'infer',
                                                   vars(args))

            predictor_model = models.get_model(
                "AttentionLSTM", predictor_infer_config, mode='infer')
            predictor_infer_config['MODEL']['topk'] = args.topk
            predictor_model.build_input(use_dataloader=False)
            predictor_model.build_model()
            predictor_feeds = predictor_model.feeds()
            predictor_outputs = predictor_model.outputs()

            exe.run(predictor_startup_prog)

            logger.info('load lstm weights from {}'.format(
                args.predictor_weights))
            predictor_model.load_test_weights(exe, args.predictor_weights,
                                              predictor_main_prog)

            predictor_feeder = fluid.DataFeeder(
                place=place, feed_list=predictor_feeds)
            predictor_fetch_list = predictor_model.fetches()

            predictor_metrics = get_metrics("AttentionLSTM".upper(), 'infer',
                                            predictor_infer_config)
            predictor_metrics.reset()

            for idx, data in enumerate(predictor_feed_list):
                file_id = data[1]
                predictor_feed_data = data[0]
                final_outs = exe.run(
                    fetch_list=predictor_fetch_list,
                    feed=predictor_feeder.feed(predictor_feed_data))
                logger.info(
                    '=======[Stage 2 Sample {} ] AttentionLSTM predict finished========'
                    .format(idx))
                final_result_list = [item for item in final_outs] + [file_id]

                predictor_metrics.accumulate(final_result_list)
            res_list = predictor_metrics.finalize_and_log_out(
                savedir=args.save_dir, label_file=args.label_file)
            return res_list
