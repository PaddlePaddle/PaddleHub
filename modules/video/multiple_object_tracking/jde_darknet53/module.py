# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import glob
import argparse

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Tracker
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.logger import setup_logger
import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, runnable
import cv2

from .tracker import StreamTracker

logger = setup_logger('Predict')


@moduleinfo(
    name="jde_darknet53",
    type="CV/multiple_object_tracking",
    author="paddlepaddle",
    author_email="",
    summary="JDE is a joint detection and appearance embedding model for multiple object tracking.",
    version="1.0.0")
class JDETracker_1088x608:
    def __init__(self):
        self.pretrained_model = os.path.join(self.directory, "jde_darknet53_30e_1088x608")

    def tracking(self, video_stream, output_dir='mot_result', visualization=True, draw_threshold=0.5, use_gpu=False):
        '''
        Track a video, and save the prediction results into output_dir, if visualization is set as True.

        video_stream: the video path
        output_dir: specify the dir to save the results
        visualization: if True, save the results as a video, otherwise not.
        draw_threshold: the threshold for the prediction results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        '''
        self.video_stream = video_stream
        self.output_dir = output_dir
        self.visualization = visualization
        self.draw_threshold = draw_threshold
        self.use_gpu = use_gpu

        cfg = load_config(os.path.join(self.directory, 'config', 'jde_darknet53_30e_1088x608.yml'))
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
            model_type='JDE',
            visualization=visualization,
            draw_threshold=draw_threshold)

    def stream_mode(self, output_dir='mot_result', visualization=True, draw_threshold=0.5, use_gpu=False):
        '''
        Entering the stream mode enables image stream prediction. Users can predict the images like a stream and save the results to a video.

        output_dir: specify the dir to save the results
        visualization: if True, save the results as a video, otherwise not.
        draw_threshold: the threshold for the prediction results
        use_gpu: if True, use gpu to perform the computation, otherwise cpu.
        '''
        self.output_dir = output_dir
        self.visualization = visualization
        self.draw_threshold = draw_threshold
        self.use_gpu = use_gpu

        cfg = load_config(os.path.join(self.directory, 'config', 'jde_darknet53_30e_1088x608.yml'))
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
            model_type='JDE',
            visualization=self.visualization,
            draw_threshold=self.draw_threshold)
        next(self.tracker_generator)

    def __exit__(self, exc_type, exc_value, traceback):
        seq = 'inputimages'
        save_dir = os.path.join(self.output_dir, 'mot_outputs', seq) if self.visualization else None
        if self.visualization:
            #### Save using ffmpeg
            #output_video_path = os.path.join(save_dir, '..', '{}_vis.mp4'.format(seq))
            #cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(
            #    save_dir, output_video_path)
            #os.system(cmd_str)
            #### Save using opencv
            output_video_path = os.path.join(save_dir, '..', '{}_vis.avi'.format(seq))
            imgnames = glob.glob(os.path.join(save_dir, '*.jpg'))
            if len(imgnames) == 0:
                logger.info('No output images to save for video')
                return
            img = cv2.imread(os.path.join(save_dir, '00000.jpg'))
            video_writer = cv2.VideoWriter(
                output_video_path,
                apiPreference=0,
                fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                fps=30,
                frameSize=(img.shape[1], img.shape[0]))
            for i in range(len(imgnames)):
                imgpath = os.path.join(save_dir, '{:05d}.jpg'.format(i))
                img = cv2.imread(imgpath)
                video_writer.write(img)
            video_writer.release()
            logger.info('Save video in {}'.format(output_video_path))

    def predict(self, images: list = []):
        '''
        Predict the images. This method should called in stream_mode.

        images: the image list used for prediction.

        Example:
        tracker = hub.Module('fairmot_dla34')
        with tracker.stream_mode(output_dir='image_stream_output', visualization=True, draw_threshold=0.5, use_gpu=True):
            tracker.predict([images])
        '''
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
        seq = os.path.splitext(os.path.basename(self.video_stream))[0]
        save_dir = os.path.join(self.output_dir, 'mot_outputs', seq) if self.visualization else None
        if self.visualization:
            #### Save using ffmpeg
            #output_video_path = os.path.join(save_dir, '..', '{}_vis.mp4'.format(seq))
            #cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(
            #    save_dir, output_video_path)
            #os.system(cmd_str)
            #### Save using opencv
            output_video_path = os.path.join(save_dir, '..', '{}_vis.avi'.format(seq))
            imgnames = glob.glob(os.path.join(save_dir, '*.jpg'))
            if len(imgnames) == 0:
                logger.info('No output images to save for video')
                return
            img = cv2.imread(os.path.join(save_dir, '00000.jpg'))
            video_writer = cv2.VideoWriter(
                output_video_path,
                apiPreference=0,
                fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                fps=30,
                frameSize=(img.shape[1], img.shape[0]))
            for i in range(len(imgnames)):
                imgpath = os.path.join(save_dir, '{:05d}.jpg'.format(i))
                img = cv2.imread(imgpath)
                video_writer.write(img)
            video_writer.release()
            logger.info('Save video in {}'.format(output_video_path))
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
            '--visualization', action='store_true', help="whether to save output as images.")
        self.arg_config_group.add_argument(
            "--draw_threshold", type=float, default=0.5, help="Threshold to reserve the result for visualization.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--video_stream', type=str, help="path to video stream, can be a video file or stream device number.")
