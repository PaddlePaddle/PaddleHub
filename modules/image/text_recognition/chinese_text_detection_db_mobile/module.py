from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from PIL import Image

from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving
from paddlehub.utils.log import logger


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if data is None:
        buf = BytesIO()
        image_decode = base64.b64decode(b64str.encode('utf8'))
        image = BytesIO(image_decode)
        im = Image.open(image)
        rgb = im.convert('RGB')
        rgb.save(buf, 'jpeg')
        buf.seek(0)
        image_bytes = buf.read()
        data_base64 = str(base64.b64encode(image_bytes), encoding="utf-8")
        image_decode = base64.b64decode(data_base64)
        img_array = np.frombuffer(image_decode, np.uint8)
        data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return data


@moduleinfo(
    name="chinese_text_detection_db_mobile",
    version="1.1.0",
    summary=
    "The module aims to detect chinese text position in the image, which is based on differentiable_binarization algorithm.",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class ChineseTextDetectionDB:

    def __init__(self, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, 'inference_model', 'model')
        self.enable_mkldnn = enable_mkldnn

        self._set_config()

    def check_requirements(self):
        try:
            import shapely, pyclipper
        except:
            raise ImportError(
                'This module requires the shapely, pyclipper tools. The running environment does not meet the requirements. Please install the two packages.'
            )

    def _set_config(self):
        """
        predictor config setting
        """
        model_file_path = self.pretrained_model_path + '.pdmodel'
        params_file_path = self.pretrained_model_path + '.pdiparams'

        config = Config(model_file_path, params_file_path)
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
            config.set_cpu_math_library_num_threads(6)
            if self.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()

        config.disable_glog_info()

        # use zero copy
        config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
        config.switch_use_feed_fetch_ops(False)
        self.predictor = create_predictor(config)
        input_names = self.predictor.get_input_names()
        self.input_tensor = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        self.output_tensors = []
        for output_name in output_names:
            output_tensor = self.predictor.get_output_handle(output_name)
            self.output_tensors.append(output_tensor)

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

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def detect_text(self,
                    images=[],
                    paths=[],
                    use_gpu=False,
                    output_dir='detection_result',
                    visualization=False,
                    box_thresh=0.5):
        """
        Get the text box in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
            use_gpu (bool): Whether to use gpu. Default false.
            output_dir (str): The directory to store output images.
            visualization (bool): Whether to save image or not.
            box_thresh(float): the threshold of the detected text box's confidence
        Returns:
            res (list): The result of text detection box and save path of images.
        """
        self.check_requirements()

        from chinese_text_detection_db_mobile.processor import DBProcessTest, DBPostProcess, draw_boxes, get_image_ext

        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        preprocessor = DBProcessTest(params={'max_side_len': 960})
        postprocessor = DBPostProcess(params={
            'thresh': 0.3,
            'box_thresh': box_thresh,
            'max_candidates': 1000,
            'unclip_ratio': 1.6
        })

        all_imgs = []
        all_ratios = []
        all_results = []
        for original_image in predicted_data:
            ori_im = original_image.copy()
            im, ratio_list = preprocessor(original_image)
            res = {'save_path': ''}
            if im is None:
                res['data'] = []

            else:
                im = im.copy()
                self.input_tensor.copy_from_cpu(im)
                self.predictor.run()

                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)

                outs_dict = {}
                outs_dict['maps'] = outputs[0]

                # data_out = self.output_tensors[0].copy_to_cpu()
                dt_boxes_list = postprocessor(outs_dict, [ratio_list])
                dt_boxes = dt_boxes_list[0]
                boxes = self.filter_tag_det_res(dt_boxes_list[0], original_image.shape)
                res['data'] = boxes.astype(np.int64).tolist()

                all_imgs.append(im)
                all_ratios.append(ratio_list)
                if visualization:
                    img = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                    draw_img = draw_boxes(img, boxes)
                    draw_img = np.array(draw_img)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    ext = get_image_ext(original_image)
                    saved_name = 'ndarray_{}{}'.format(time.time(), ext)
                    cv2.imwrite(os.path.join(output_dir, saved_name), draw_img[:, :, ::-1])
                    res['save_path'] = os.path.join(output_dir, saved_name)

            all_results.append(res)

        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.detect_text(images=images_decode, **kwargs)
        return results

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
        results = self.detect_text(paths=[args.input_path],
                                   use_gpu=args.use_gpu,
                                   output_dir=args.output_dir,
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
                                           default='detection_result',
                                           help="The directory to save output images.")
        self.arg_config_group.add_argument('--visualization',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--input_path', type=str, default=None, help="diretory to image")
