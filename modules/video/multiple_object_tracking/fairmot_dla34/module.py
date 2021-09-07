# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import sys
import signal
import argparse
# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
from paddle.distributed import ParallelEnv
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Tracker
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.logger import setup_logger
import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, runnable

from .tracker import StreamTracker

logger = setup_logger('Predict')
parent_path = os.path.abspath(os.path.join(__file__, '..'))
if parent_path not in sys.path:
    sys.path.append(parent_path)


@moduleinfo(
    name="fairmot_dla34",
    type="CV/multiple_object_tracking",
    author="paddlepaddle",
    author_email="",
    summary="Fairmot is a model for multiple object tracking.",
    version="1.0.0")
class FairmotTracker_1088x608(hub.Module):
    def _initialize(self):
        self.pretrained_model = os.path.join(self.directory, "fairmot_dla34_30e_1088x608")

    def tracking(self, video_stream, output_dir='mot_result', visualization=True, draw_threshold=0.5, use_gpu=False):
        '''

        '''
        self.video_stream = video_stream
        self.output_dir = output_dir
        self.visualization = visualization
        self.draw_threshold = draw_threshold
        self.use_gpu = use_gpu

        cfg = load_config(os.path.join(self.directory, 'config', 'fairmot_dla34_30e_1088x608.yml'))
        check_config(cfg)

        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)

        paddle.disable_static()
        tracker = StreamTracker(cfg, mode='test')

        # load weights
        tracker.load_weights_jde(self.pretrained_model)
        signal.signal(signal.SIGINT, self.signalhandler)
        # inference
        tracker.videostream_predict(
            video_stream=video_stream,
            output_dir=output_dir,
            data_type='mot',
            model_type='FairMOT',
            visualization=visualization,
            draw_threshold=draw_threshold)

    def stream_mode(self, output_dir='mot_result', visualization=True, draw_threshold=0.5, use_gpu=False):
        self.output_dir = output_dir
        self.visualization = visualization
        self.draw_threshold = draw_threshold
        self.use_gpu = use_gpu

        cfg = load_config(os.path.join(self.directory, 'config', 'fairmot_dla34_30e_1088x608.yml'))
        check_config(cfg)

        place = 'gpu:0' if use_gpu else 'cpu'
        place = paddle.set_device(place)

        paddle.disable_static()
        self.tracker = StreamTracker(cfg, mode='test')

        # load weights
        self.tracker.load_weights_jde(self.pretrained_model)
        signal.signal(signal.SIGINT, self.signalhandler)
        return self

    def __enter__(self):
        self.tracker_generator = self.tracker.imagestream_predict(
            self.output_dir,
            data_type='mot',
            model_type='FairMOT',
            visualization=self.visualization,
            draw_threshold=self.draw_threshold)

    def __exit__(self):
        seq = 'inputimages'
        save_dir = os.path.join(self.output_dir, 'mot_outputs', seq) if self.args.visualization else None
        if self.args.visualization:
            output_video_path = os.path.join(save_dir, '..', '{}_vis.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(
                save_dir, output_video_path)
            os.system(cmd_str)

    def predict(self, images: list = []):
        length = len(images)
        if length == 0:
            print('No images provided.')
            return
        for image in images:
            self.tracker.dataset.add_image(image)
            try:
                results = next(self.tracker_generator)
            except StopIteration as e:
                return

        return results[-length:]

    @runnable
    def run_cmd(self, argvs: list):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        self.args = self.parser.parse_args(argvs)
        self.tracking(
            video_stream=self.args.video_stream,
            output_dir=self.args.output_dir,
            visualization=self.args.visualization,
            draw_threshold=self.args.draw_threshold,
            use_gpu=self.args.use_gpu,
        )

    def signalhandler(self, signum, frame):
        seq = os.path.splitext(os.path.basename(self.args.video_stream))[0]
        save_dir = os.path.join(self.args.output_dir, 'mot_outputs', seq) if self.args.visualization else None
        if self.args.visualization:
            output_video_path = os.path.join(save_dir, '..', '{}_vis.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(
                save_dir, output_video_path)
            os.system(cmd_str)
            print('Program Interrupted! Save video in {}'.format(output_video_path))
            exit(0)

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', action='store_true', help="use GPU or not")

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='mot_result', help='Directory name for output tracking results.')
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")
        self.arg_config_group.add_argument(
            "--draw_threshold", type=float, default=0.5, help="Threshold to reserve the result for visualization.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--video_stream', type=str, help="path to video stream, can be a video file or stream device number.")
