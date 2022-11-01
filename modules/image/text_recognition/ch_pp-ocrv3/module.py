# -*- coding:utf-8 -*-
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import argparse
import ast
import copy
import math
import os
import time

import cv2
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.inference as paddle_infer
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor
from paddle.fluid.core import PaddleTensor
from PIL import Image

import paddlehub as hub
from .character import CharacterOps
from .utils import base64_to_cv2
from .utils import draw_ocr
from .utils import get_image_ext
from .utils import sorted_boxes
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(
    name="ch_pp-ocrv3",
    version="1.0.0",
    summary="The module can recognize the chinese texts in an image. Firstly, it will detect the text box positions \
        based on the differentiable_binarization_chn module. Then it classifies the text angle and recognizes the chinese texts. ",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class ChPPOCRv3(hub.Module):

    def _initialize(self, text_detector_module=None, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        self.character_dict_path = os.path.join(self.directory, 'assets', 'ppocr_keys_v1.txt')
        char_ops_params = {
            'character_type': 'ch',
            'character_dict_path': self.character_dict_path,
            'loss_type': 'ctc',
            'max_text_length': 25,
            'use_space_char': True
        }
        self.char_ops = CharacterOps(char_ops_params)
        self.rec_image_shape = [3, 48, 320]
        self._text_detector_module = text_detector_module
        self.font_file = os.path.join(self.directory, 'assets', 'simfang.ttf')
        self.enable_mkldnn = enable_mkldnn

        self.rec_pretrained_model_path = os.path.join(self.directory, 'inference_model', 'ppocrv3_rec')
        self.cls_pretrained_model_path = os.path.join(self.directory, 'inference_model', 'ppocr_cls')
        self.rec_predictor, self.rec_input_tensor, self.rec_output_tensors = self._set_config(
            self.rec_pretrained_model_path)
        self.cls_predictor, self.cls_input_tensor, self.cls_output_tensors = self._set_config(
            self.cls_pretrained_model_path)

    def _set_config(self, pretrained_model_path):
        """
        predictor config path
        """
        model_file_path = pretrained_model_path + '.pdmodel'
        params_file_path = pretrained_model_path + '.pdiparams'

        config = paddle_infer.Config(model_file_path, params_file_path)
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False

        if use_gpu:
            config.enable_use_gpu(8000, 0)
        else:
            config.disable_gpu()
            if self.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()

        config.disable_glog_info()
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)

        predictor = paddle_infer.create_predictor(config)

        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])
        output_names = predictor.get_output_names()
        output_handles = []
        for output_name in output_names:
            output_handle = predictor.get_output_handle(output_name)
            output_handles.append(output_handle)

        return predictor, input_handle, output_handles

    @property
    def text_detector_module(self):
        """
        text detect module
        """
        if not self._text_detector_module:
            self._text_detector_module = hub.Module(name='ch_pp-ocrv3_det',
                                                    enable_mkldnn=self.enable_mkldnn,
                                                    version='1.0.0')
        return self._text_detector_module

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(img,
                                      M, (img_crop_width, img_crop_height),
                                      borderMode=cv2.BORDER_REPLICATE,
                                      flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def resize_norm_img_rec(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_cls(self, img):
        cls_image_shape = [3, 48, 192]
        imgC, imgH, imgW = cls_image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def recognize_text(self,
                       images=[],
                       paths=[],
                       use_gpu=False,
                       output_dir='ocr_result',
                       visualization=False,
                       box_thresh=0.5,
                       text_thresh=0.5,
                       angle_classification_thresh=0.9,
                       det_db_unclip_ratio=1.5):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            use_gpu (bool): Whether to use gpu.
            batch_size(int): the program deals once with one
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
            box_thresh(float): the threshold of the detected text box's confidence
            text_thresh(float): the threshold of the chinese text recognition confidence
            angle_classification_thresh(float): the threshold of the angle classification confidence
            det_db_unclip_ratio(float): unclip ratio for post processing in DB detection.
        Returns:
            res (list): The result of chinese texts and save path of images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )

        self.use_gpu = use_gpu

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        detection_results = self.text_detector_module.detect_text(images=predicted_data,
                                                                  use_gpu=self.use_gpu,
                                                                  box_thresh=box_thresh,
                                                                  det_db_unclip_ratio=det_db_unclip_ratio)

        boxes = [np.array(item['data']).astype(np.float32) for item in detection_results]
        all_results = []
        for index, img_boxes in enumerate(boxes):
            original_image = predicted_data[index].copy()
            result = {'save_path': ''}
            if img_boxes.size == 0:
                result['data'] = []
            else:
                img_crop_list = []
                boxes = sorted_boxes(img_boxes)
                for num_box in range(len(boxes)):
                    tmp_box = copy.deepcopy(boxes[num_box])
                    img_crop = self.get_rotate_crop_image(original_image, tmp_box)
                    img_crop_list.append(img_crop)
                img_crop_list, angle_list = self._classify_text(img_crop_list,
                                                                angle_classification_thresh=angle_classification_thresh)
                rec_results = self._recognize_text(img_crop_list)

                # if the recognized text confidence score is lower than text_thresh, then drop it
                rec_res_final = []
                for index, res in enumerate(rec_results):
                    text, score = res
                    if score >= text_thresh:
                        rec_res_final.append({
                            'text': text,
                            'confidence': float(score),
                            'text_box_position': boxes[index].astype(np.int).tolist()
                        })
                result['data'] = rec_res_final

                if visualization and result['data']:
                    result['save_path'] = self.save_result_image(original_image, boxes, rec_results, output_dir,
                                                                 text_thresh)
            all_results.append(result)

        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.recognize_text(images_decode, **kwargs)
        return results

    def save_result_image(
        self,
        original_image,
        detection_boxes,
        rec_results,
        output_dir='ocr_result',
        text_thresh=0.5,
    ):
        image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        txts = [item[0] for item in rec_results]
        scores = [item[1] for item in rec_results]
        draw_img = draw_ocr(image,
                            detection_boxes,
                            txts,
                            scores,
                            font_file=self.font_file,
                            draw_txt=True,
                            drop_score=text_thresh)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ext = get_image_ext(original_image)
        saved_name = 'ndarray_{}{}'.format(time.time(), ext)
        save_file_path = os.path.join(output_dir, saved_name)
        cv2.imwrite(save_file_path, draw_img[:, :, ::-1])
        return save_file_path

    def _classify_text(self, image_list, angle_classification_thresh=0.9):
        img_list = copy.deepcopy(image_list)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the cls process
        indices = np.argsort(np.array(width_list))

        cls_res = [['', 0.0]] * img_num
        batch_num = 6
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img_cls(img_list[indices[ino]])
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            self.cls_input_tensor.copy_from_cpu(norm_img_batch)
            self.cls_predictor.run()

            prob_out = self.cls_output_tensors[0].copy_to_cpu()
            ## post process
            label_list = ['0', '180']
            pred_idxs = prob_out.argmax(axis=1)
            cls_result = [(label_list[idx], prob_out[i, idx]) for i, idx in enumerate(pred_idxs)]
            for rno in range(len(cls_result)):
                label, score = cls_result[rno]
                cls_res[indices[beg_img_no + rno]] = [label, score]
                if '180' in label and score > angle_classification_thresh:
                    img_list[indices[beg_img_no + rno]] = cv2.rotate(img_list[indices[beg_img_no + rno]], 1)
        return img_list, cls_res

    def _recognize_text(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        rec_res = [['', 0.0]] * img_num
        batch_num = 6
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img_rec(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch, axis=0)
            norm_img_batch = norm_img_batch.copy()

            self.rec_input_tensor.copy_from_cpu(norm_img_batch)
            self.rec_predictor.run()

            ##
            outputs = []
            for output_tensor in self.rec_output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]
            if isinstance(preds, tuple) or isinstance(preds, list):
                preds = preds[-1]
            if isinstance(preds, paddle.Tensor):
                preds = preds.numpy()
            preds_idx = preds.argmax(axis=2)
            preds_prob = preds.max(axis=2)
            rec_result = self.char_ops.decode(preds_idx, preds_prob, is_remove_duplicate=True)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(description="Run the %s module." % self.name,
                                              prog='hub run %s' % self.name,
                                              usage='%(prog)s',
                                              add_help=True)

        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)
        results = self.recognize_text(paths=[args.input_path],
                                      use_gpu=args.use_gpu,
                                      output_dir=args.output_dir,
                                      det_db_unclip_ratio=args.det_db_unclip_ratio,
                                      visualization=args.visualization)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not")
        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='ocr_result',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether to save output as images.")
        self.arg_config_group.add_argument('--det_db_unclip_ratio',
                                           type=float,
                                           default=1.5,
                                           help="unclip ratio for post processing in DB detection.")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--input_path', type=str, default=None, help="diretory to image")
