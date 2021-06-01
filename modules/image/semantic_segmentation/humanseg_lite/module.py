# -*- coding:utf-8 -*-
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
import os
import os.path as osp
import argparse

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from humanseg_lite.processor import postprocess, base64_to_cv2, cv2_to_base64, check_dir
from humanseg_lite.data_feed import reader, preprocess_v
from humanseg_lite.optimal import postprocess_v, threshold_mask


@moduleinfo(
    name="humanseg_lite",
    type="CV/semantic_segmentation",
    author="paddlepaddle",
    author_email="",
    summary="humanseg_lite is a semantic segmentation model.",
    version="1.1.0")
class ShufflenetHumanSeg(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "humanseg_lite_inference")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        self.model_file_path = os.path.join(self.default_pretrained_model_path, '__model__')
        self.params_file_path = os.path.join(self.default_pretrained_model_path, '__params__')
        cpu_config = AnalysisConfig(self.model_file_path, self.params_file_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False

        if use_gpu:
            gpu_config = AnalysisConfig(self.model_file_path, self.params_file_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def segment(self,
                images=None,
                paths=None,
                batch_size=1,
                use_gpu=False,
                visualization=False,
                output_dir='humanseg_lite_output'):
        """
        API for human segmentation.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C], the color space is BGR.
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            visualization (bool): Whether to save image or not.
            output_dir (str): The path to store output images.

        Returns:
            res (list[dict]): each element in the list is a dict, the keys and values are:
                save_path (str, optional): the path to save images. (Exists only if visualization is True)
                data (numpy.ndarray): data of post processed image.
        """

        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )

        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])
            batch_image = PaddleTensor(batch_image.copy())
            output = self.gpu_predictor.run([batch_image]) if use_gpu else self.cpu_predictor.run([batch_image])
            output = output[1].as_ndarray()
            output = np.expand_dims(output[:, 1, :, :], axis=1)
            # postprocess one by one
            for i in range(len(batch_data)):
                out = postprocess(
                    data_out=output[i],
                    org_im=batch_data[i]['org_im'],
                    org_im_shape=batch_data[i]['org_im_shape'],
                    org_im_path=batch_data[i]['org_im_path'],
                    output_dir=output_dir,
                    visualization=visualization)
                res.append(out)
        return res

    def video_stream_segment(self, frame_org, frame_id, prev_gray, prev_cfd, use_gpu=False):
        """
        API for human video segmentation.

        Args:
           frame_org (numpy.ndarray): frame data, shape of each is [H, W, C], the color space is BGR.
           frame_id (int): index of the frame to be decoded.
           prev_gray (numpy.ndarray): gray scale image of last frame, shape of each is [H, W]
           prev_cfd (numpy.ndarray): fusion image from optical flow image and segment result, shape of each is [H, W]
           use_gpu (bool): Whether to use gpu.

        Returns:
            img_matting (numpy.ndarray): data of segmentation mask.
            cur_gray (numpy.ndarray): gray scale image of current frame, shape of each is [H, W]
            optflow_map (numpy.ndarray): optical flow image of current frame, shape of each is [H, W]

        """
        resize_h = 192
        resize_w = 192
        is_init = True
        width = int(frame_org.shape[0])
        height = int(frame_org.shape[1])
        disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        frame = preprocess_v(frame_org, resize_w, resize_h)
        image = PaddleTensor(np.array([frame.copy()]))
        output = self.gpu_predictor.run([image]) if use_gpu else self.cpu_predictor.run([image])
        score_map = output[1].as_ndarray()
        frame = np.transpose(frame, axes=[1, 2, 0])
        score_map = np.transpose(np.squeeze(score_map, 0), axes=[1, 2, 0])
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
        score_map = 255 * score_map[:, :, 1]
        if frame_id == 1:
            prev_gray = np.zeros((resize_h, resize_w), np.uint8)
            prev_cfd = np.zeros((resize_h, resize_w), np.float32)
            optflow_map = postprocess_v(cur_gray, score_map, prev_gray, prev_cfd, disflow, is_init)
        else:
            optflow_map = postprocess_v(cur_gray, score_map, prev_gray, prev_cfd, disflow, is_init)

        optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
        optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
        img_matting = cv2.resize(optflow_map, (height, width), cv2.INTER_LINEAR)

        return [img_matting, cur_gray, optflow_map]

    def video_segment(self, video_path=None, use_gpu=False, save_dir='humanseg_lite_video_result'):
        """
        API for human video segmentation.

        Args:
           video_path (str): The path to take the video under preprocess. If video_path is None, it will capture
           the vedio from your camera.
           use_gpu (bool): Whether to use gpu.
           save_dir (str): The path to store output video.

        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError("Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. "
                                   "If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id.")

        resize_h = 192
        resize_w = 192
        if not video_path:
            cap_video = cv2.VideoCapture(0)
        else:
            cap_video = cv2.VideoCapture(video_path)

        if not cap_video.isOpened():
            raise IOError("Error opening video stream or file, "
                          "--video_path whether existing: {}"
                          " or camera whether working".format(video_path))

        width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        prev_gray = np.zeros((resize_h, resize_w), np.uint8)
        prev_cfd = np.zeros((resize_h, resize_w), np.float32)
        is_init = True
        fps = cap_video.get(cv2.CAP_PROP_FPS)

        if video_path is not None:
            print('Please wait. It is computing......')
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            save_path = osp.join(save_dir, 'result' + '.avi')
            cap_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

            while cap_video.isOpened():
                ret, frame_org = cap_video.read()
                if ret:
                    frame = preprocess_v(frame_org, resize_w, resize_h)
                    image = PaddleTensor(np.array([frame.copy()]))
                    output = self.gpu_predictor.run([image]) if use_gpu else self.cpu_predictor.run([image])
                    score_map = output[1].as_ndarray()
                    frame = np.transpose(frame, axes=[1, 2, 0])
                    score_map = np.transpose(np.squeeze(score_map, 0), axes=[1, 2, 0])
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                    score_map = 255 * score_map[:, :, 1]
                    optflow_map = postprocess_v(cur_gray, score_map, prev_gray, prev_cfd, disflow, is_init)
                    prev_gray = cur_gray.copy()
                    prev_cfd = optflow_map.copy()

                    optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                    optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                    img_matting = cv2.resize(optflow_map, (width, height), cv2.INTER_LINEAR)
                    img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
                    bg_im = np.ones_like(img_matting) * 255
                    comb = (img_matting * frame_org + (1 - img_matting) * bg_im).astype(np.uint8)
                    cap_out.write(comb)
                else:
                    break
            cap_video.release()
            cap_out.release()
        else:
            while cap_video.isOpened():
                ret, frame_org = cap_video.read()
                if ret:
                    frame = preprocess_v(frame_org, resize_w, resize_h)
                    image = PaddleTensor(np.array([frame.copy()]))
                    output = self.gpu_predictor.run([image]) if use_gpu else self.cpu_predictor.run([image])
                    score_map = output[1].as_ndarray()
                    frame = np.transpose(frame, axes=[1, 2, 0])
                    score_map = np.transpose(np.squeeze(score_map, 0), axes=[1, 2, 0])
                    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cur_gray = cv2.resize(cur_gray, (resize_w, resize_h))
                    score_map = 255 * score_map[:, :, 1]
                    optflow_map = postprocess_v(cur_gray, score_map, prev_gray, prev_cfd, disflow, is_init)
                    prev_gray = cur_gray.copy()
                    prev_cfd = optflow_map.copy()
                    optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
                    optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
                    img_matting = cv2.resize(optflow_map, (width, height), cv2.INTER_LINEAR)
                    img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
                    bg_im = np.ones_like(img_matting) * 255
                    comb = (img_matting * frame_org + (1 - img_matting) * bg_im).astype(np.uint8)
                    cv2.imshow('HumanSegmentation', comb)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap_video.release()

    def save_inference_model(self,
                             dirname='humanseg_lite_model',
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path,
            model_filename=model_filename,
            params_filename=params_filename,
            executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=program,
            executor=exe,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            model_filename=model_filename,
            params_filename=params_filename)

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.segment(images=images_decode, **kwargs)
        results = [{'data': cv2_to_base64(result['data'])} for result in results]
        return results

    @runnable
    def run_cmd(self, argvs):
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
        args = self.parser.parse_args(argvs)
        results = self.segment(
            paths=[args.input_path],
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
        if args.save_dir is not None:
            check_dir(args.save_dir)
            self.save_inference_model(args.save_dir)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu', type=ast.literal_eval, default=False, help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='humanseg_lite_output', help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--save_dir', type=str, default='humanseg_lite_model', help="The directory to save model.")
        self.arg_config_group.add_argument(
            '--visualization', type=ast.literal_eval, default=False, help="whether to save output as images.")
        self.arg_config_group.add_argument('--batch_size', type=ast.literal_eval, default=1, help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


if __name__ == "__main__":
    m = ShufflenetHumanSeg()
    #shuffle.video_segment()
    img = cv2.imread('photo.jpg')
    # res = m.segment(images=[img], visualization=True)
    # print(res[0]['data'])
    # m.video_segment('')
    cap_video = cv2.VideoCapture('video_test.mp4')
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    save_path = 'result_frame.avi'
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
    prev_gray = None
    prev_cfd = None
    while cap_video.isOpened():
        ret, frame_org = cap_video.read()
        if ret:
            [img_matting, prev_gray, prev_cfd] = m.video_stream_segment(
                frame_org=frame_org, frame_id=cap_video.get(1), prev_gray=prev_gray, prev_cfd=prev_cfd)
            img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
            bg_im = np.ones_like(img_matting) * 255
            comb = (img_matting * frame_org + (1 - img_matting) * bg_im).astype(np.uint8)
            cap_out.write(comb)
        else:
            break

    cap_video.release()
    cap_out.release()
