# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import copy
import math
import os
import time

from paddle.fluid.core import AnalysisConfig, create_paddle_predictor, PaddleTensor
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
from PIL import Image
import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

from chinese_ocr_db_crnn_mobile.character import CharacterOps
from chinese_ocr_db_crnn_mobile.utils import base64_to_cv2, draw_ocr, get_image_ext, sorted_boxes


@moduleinfo(
    name="chinese_ocr_db_crnn_mobile",
    version="1.0.1",
    summary=
    "The module can recognize the chinese texts in an image. Firstly, it will detect the text box positions based on the differentiable_binarization_chn module. Then it recognizes the chinese texts. ",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class ChineseOCRDBCRNN(hub.Module):
    def _initialize(self, text_detector_module=None):
        """
        initialize with the necessary elements
        """
        self.character_dict_path = os.path.join(self.directory, 'assets',
                                                'ppocr_keys_v1.txt')
        char_ops_params = {
            'character_type': 'ch',
            'character_dict_path': self.character_dict_path,
            'loss_type': 'ctc'
        }
        self.char_ops = CharacterOps(char_ops_params)
        self.rec_image_shape = [3, 32, 320]
        self._text_detector_module = text_detector_module
        self.font_file = os.path.join(self.directory, 'assets', 'simfang.ttf')
        self.pretrained_model_path = os.path.join(self.directory,
                                                  'inference_model')
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        model_file_path = os.path.join(self.pretrained_model_path, 'model')
        params_file_path = os.path.join(self.pretrained_model_path, 'params')

        config = AnalysisConfig(model_file_path, params_file_path)
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

        config.disable_glog_info()

        # use zero copy
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        self.predictor = create_paddle_predictor(config)
        input_names = self.predictor.get_input_names()
        self.input_tensor = self.predictor.get_input_tensor(input_names[0])
        output_names = self.predictor.get_output_names()
        self.output_tensors = []
        for output_name in output_names:
            output_tensor = self.predictor.get_output_tensor(output_name)
            self.output_tensors.append(output_tensor)

    @property
    def text_detector_module(self):
        """
        text detect module
        """
        if not self._text_detector_module:
            self._text_detector_module = hub.Module(
                name='chinese_text_detection_db_mobile')
        return self._text_detector_module

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def get_rotate_crop_image(self, img, points):
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        img_crop_width = int(np.linalg.norm(points[0] - points[1]))
        img_crop_height = int(np.linalg.norm(points[0] - points[3]))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],\
            [img_crop_width, img_crop_height], [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(32 * max_wh_ratio)
        h = img.shape[0]
        w = img.shape[1]
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

    def recognize_text(self,
                       images=[],
                       paths=[],
                       use_gpu=False,
                       output_dir='ocr_result',
                       visualization=False,
                       box_thresh=0.5,
                       text_thresh=0.5):
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
            text_thresh(float): the threshold of the recognize chinese texts' confidence
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

        detection_results = self.text_detector_module.detect_text(
            images=predicted_data, use_gpu=self.use_gpu, box_thresh=box_thresh)
        boxes = [
            np.array(item['data']).astype(np.float32)
            for item in detection_results
        ]
        all_results = []
        for index, img_boxes in enumerate(boxes):
            original_image = predicted_data[index].copy()
            result = {'save_path': ''}
            if img_boxes is None:
                result['data'] = []
            else:
                img_crop_list = []
                boxes = sorted_boxes(img_boxes)
                for num_box in range(len(boxes)):
                    tmp_box = copy.deepcopy(boxes[num_box])
                    img_crop = self.get_rotate_crop_image(
                        original_image, tmp_box)
                    img_crop_list.append(img_crop)

                rec_results = self._recognize_text(img_crop_list)
                # if the recognized text confidence score is lower than text_thresh, then drop it
                rec_res_final = []
                for index, res in enumerate(rec_results):
                    text, score = res
                    if score >= text_thresh:
                        rec_res_final.append({
                            'text':
                            text,
                            'confidence':
                            float(score),
                            'text_box_position':
                            boxes[index].astype(np.int).tolist()
                        })
                result['data'] = rec_res_final

                if visualization and result['data']:
                    result['save_path'] = self.save_result_image(
                        original_image, boxes, rec_results, output_dir,
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

    def save_result_image(self,
                          original_image,
                          detection_boxes,
                          rec_results,
                          output_dir='ocr_result',
                          text_thresh=0.5):
        image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        txts = [item[0] for item in rec_results]
        scores = [item[1] for item in rec_results]
        draw_img = draw_ocr(
            image,
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

    def _recognize_text(self, image_list):
        img_num = len(image_list)
        batch_num = 30
        rec_res = []
        predict_time = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = image_list[ino].shape[0:2]
                wh_ratio = w / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(image_list[ino], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            self.input_tensor.copy_from_cpu(norm_img_batch)
            self.predictor.zero_copy_run()
            rec_idx_batch = self.output_tensors[0].copy_to_cpu()
            rec_idx_lod = self.output_tensors[0].lod()[0]
            predict_batch = self.output_tensors[1].copy_to_cpu()
            predict_lod = self.output_tensors[1].lod()[0]

            for rno in range(len(rec_idx_lod) - 1):
                beg = rec_idx_lod[rno]
                end = rec_idx_lod[rno + 1]
                rec_idx_tmp = rec_idx_batch[beg:end, 0]
                preds_text = self.char_ops.decode(rec_idx_tmp)
                beg = predict_lod[rno]
                end = predict_lod[rno + 1]
                probs = predict_batch[beg:end, :]
                ind = np.argmax(probs, axis=1)
                blank = probs.shape[1]
                valid_ind = np.where(ind != (blank - 1))[0]
                score = np.mean(probs[valid_ind, ind[valid_ind]])
                rec_res.append([preds_text, score])

        return rec_res

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        detector_dir = os.path.join(dirname, 'text_detector')
        recognizer_dir = os.path.join(dirname, 'text_recognizer')
        self._save_detector_model(detector_dir, model_filename, params_filename,
                                  combined)
        self._save_recognizer_model(recognizer_dir, model_filename,
                                    params_filename, combined)
        logger.info("The inference model has been saved in the path {}".format(
            os.path.realpath(dirname)))

    def _save_detector_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True):
        self.text_detector_module.save_inference_model(
            dirname, model_filename, params_filename, combined)

    def _save_recognizer_model(self,
                               dirname,
                               model_filename=None,
                               params_filename=None,
                               combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        model_file_path = os.path.join(self.pretrained_model_path, 'model')
        params_file_path = os.path.join(self.pretrained_model_path, 'params')
        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.pretrained_model_path,
            model_filename=model_file_path,
            params_filename=params_file_path,
            executor=exe)

        fluid.io.save_inference_model(
            dirname=dirname,
            main_program=program,
            executor=exe,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            model_filename=model_filename,
            params_filename=params_filename)

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the %s module." % self.name,
            prog='hub run %s' % self.name,
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)
        results = self.recognize_text(
            paths=[args.input_path],
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default='ocr_result',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, default=None, help="diretory to image")


if __name__ == '__main__':
    ocr = ChineseOCRDBCRNN()
    image_path = [
        '/mnt/zhangxuefei/PaddleOCR/doc/imgs/11.jpg',
        '/mnt/zhangxuefei/PaddleOCR/doc/imgs/12.jpg',
        '/mnt/zhangxuefei/PaddleOCR/doc/imgs/test_image.jpg'
    ]
    res = ocr.recognize_text(paths=image_path, visualization=True)
    ocr.save_inference_model('save')
    print(res)
