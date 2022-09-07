# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
from math import ceil

import paddle
import numpy as np
import paddle.jit
import paddle.static
from paddlehub.module.module import moduleinfo, runnable, serving
from paddle.inference import Config, create_predictor
from paddlehub.utils.parser import txt_parser
from .processor import load_label_info, postprocess, base64_to_cv2
from .data_feed import test_reader, padding_minibatch


@moduleinfo(
    name="faster_rcnn_resnet50_fpn_coco2017",
    version="1.1.0",
    type="cv/object_detection",
    summary=
    "Baidu's Faster-RCNN model for object detection, whose backbone is ResNet50, processed with Feature Pyramid Networks",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNResNet50RPN:
    def __init__(self):
        # default pretrained model, Faster-RCNN with backbone ResNet50, shape of input tensor is [3, 800, 1333]
        self.default_pretrained_model_path = os.path.join(
            self.directory, "faster_rcnn_resnet50_fpn_model", "model")
        self.label_names = load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        model = self.default_pretrained_model_path+'.pdmodel'
        params = self.default_pretrained_model_path+'.pdiparams'
        cpu_config = Config(model, params)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = Config(model, params)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_predictor(gpu_config)

    def save_inference_model(self, path):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        program, feed_target_names, fetch_targets = paddle.static.load_inference_model(self.default_pretrained_model_path, exe)
        global_block = program.global_block()
        feed_vars = [global_block.var(item) for item in feed_target_names]
        paddle.static.save_inference_model(
            path,
            feed_vars=feed_vars,
            fetch_vars=fetch_targets,
            executor=exe,
            program=program
        )

    def object_detection(self,
                         paths=None,
                         images=None,
                         use_gpu=False,
                         batch_size=1,
                         output_dir='detection_result',
                         score_thresh=0.5,
                         visualization=True):
        """API of Object Detection.

        Args:
            paths (list[str]): The paths of images.
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            score_thresh (float): threshold for object detecion.

        Returns:
            res (list[dict]): The result of coco2017 detecion. keys include 'data', 'save_path', the corresponding value is:
                data (dict): the result of object detection, keys include 'left', 'top', 'right', 'bottom', 'label', 'confidence', the corresponding value is:
                    left (float): The X coordinate of the upper left corner of the bounding box;
                    top (float): The Y coordinate of the upper left corner of the bounding box;
                    right (float): The X coordinate of the lower right corner of the bounding box;
                    bottom (float): The Y coordinate of the lower right corner of the bounding box;
                    label (str): The label of detection result;
                    confidence (float): The confidence of detection result.
                save_path (str, optional): The path to save output images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Attempt to use GPU for prediction, but environment variable CUDA_VISIBLE_DEVICES was not set correctly."
                )

        paths = paths if paths else list()

        all_images = list()
        for yield_data in test_reader(paths, images):
            all_images.append(yield_data)

        images_num = len(all_images)
        loop_num = ceil(images_num / batch_size)
        res = []

        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size

            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass

            padding_image, padding_info, padding_shape = padding_minibatch(
                batch_data, coarsest_stride=32, use_padded_im_info=True)
            feed_list = [
                padding_image, padding_info, padding_shape
            ]

            predictor = self.gpu_predictor if use_gpu else self.cpu_predictor

            feed_list = [
                padding_image, padding_info, padding_shape
            ]

            input_names = predictor.get_input_names()
            
            for i, input_name in enumerate(input_names):
                data = np.asarray(feed_list[i], dtype=np.float32)
                handle = predictor.get_input_handle(input_name)
                handle.copy_from_cpu(data)
            
            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            output = postprocess(
                paths=paths,
                images=images,
                data_out=output_handle,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_dir,
                handle_id=handle_id,
                visualization=visualization)
            res += output
        return res

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
            '--batch_size',
            type=int,
            default=1,
            help="batch size for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, default=None, help="input data")

        self.arg_input_group.add_argument(
            '--input_file',
            type=str,
            default=None,
            help="file contain input data")

    def check_input_data(self, args):
        input_data = []
        if args.input_path:
            input_data = [args.input_path]
        elif args.input_file:
            if not os.path.exists(args.input_file):
                raise RuntimeError("File %s is not exist." % args.input_file)
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        return input_data

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.object_detection(images=images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        self.parser = argparse.ArgumentParser(
            description="Run the {}".format(self.name),
            prog="hub run {}".format(self.name),
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
        input_data = self.check_input_data(args)
        if len(input_data) == 0:
            self.parser.print_help()
            exit(1)
        else:
            for image_path in input_data:
                if not os.path.exists(image_path):
                    raise RuntimeError(
                        "File %s or %s is not exist." % image_path)
        return self.object_detection(
            paths=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)
