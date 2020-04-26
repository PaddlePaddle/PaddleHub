# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
from collections import OrderedDict
from functools import partial
from math import ceil

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.io.parser import txt_parser

from faster_rcnn_resnet50_fpn_coco2017.fpn import FPN
from faster_rcnn_resnet50_fpn_coco2017.resnet import ResNet, ResNetC5


@moduleinfo(
    name="faster_rcnn_resnet50_fpn_coco2017",
    version="1.0.0",
    type="cv/object_detection",
    summary=
    "Baidu's Faster-RCNN model for object detection, whose backbone is ResNet50, processed with Feature Pyramid Networks",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNResNet50RPN(hub.Module):
    def _initialize(self):
        self.faster_rcnn = hub.Module(name="faster_rcnn")
        # default pretrained model, Faster-RCNN with backbone ResNet50, shape of input tensor is [3, 800, 1333]
        self.default_pretrained_model_path = os.path.join(
            self.directory, "faster_rcnn_resnet50_fpn_model")
        self.label_names = self.faster_rcnn.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.bbox_out = None
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
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
            gpu_config = AnalysisConfig(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self,
                num_classes=81,
                trainable=True,
                pretrained=True,
                phase='train'):
        """Distill the Head Features, so as to perform transfer learning.

        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param phase: Optional Choice: 'predict', 'train'
        :type phase: str
        """
        wrapped_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(wrapped_prog, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(
                    name='image', shape=[3, 800, 1333], dtype='float32')
                # backbone
                backbone = ResNet(
                    norm_type='affine_channel',
                    depth=50,
                    feature_maps=[2, 3, 4, 5],
                    freeze_at=2)
                body_feats = backbone(image)
                # fpn: FPN
                fpn = FPN(
                    max_level=6,
                    min_level=2,
                    num_chan=256,
                    spatial_scale=[0.03125, 0.0625, 0.125, 0.25])

                # Base Class
                inputs, outputs, context_prog = self.faster_rcnn.context(
                    body_feats=body_feats,
                    fpn=fpn,
                    rpn_head=self.rpn_head(),
                    roi_extractor=self.roi_extractor(),
                    bbox_head=self.bbox_head(num_classes),
                    bbox_assigner=self.bbox_assigner(num_classes),
                    image=image,
                    trainable=trainable,
                    var_prefix='@HUB_{}@'.format(self.name),
                    phase=phase)

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                if pretrained:

                    def _if_exist(var):
                        return os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))

                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        predicate=_if_exist)
                else:
                    exe.run(startup_program)
                return inputs, outputs, context_prog

    def rpn_head(self):
        return self.faster_rcnn.FPNRPNHead(
            anchor_generator=self.faster_rcnn.AnchorGenerator(
                anchor_sizes=[32, 64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                stride=[16.0, 16.0],
                variance=[1.0, 1.0, 1.0, 1.0]),
            rpn_target_assign=self.faster_rcnn.RPNTargetAssign(
                rpn_batch_size_per_im=256,
                rpn_fg_fraction=0.5,
                rpn_negative_overlap=0.3,
                rpn_positive_overlap=0.7,
                rpn_straddle_thresh=0.0),
            train_proposal=self.faster_rcnn.GenerateProposals(
                min_size=0.0,
                nms_thresh=0.7,
                post_nms_top_n=2000,
                pre_nms_top_n=2000),
            test_proposal=self.faster_rcnn.GenerateProposals(
                min_size=0.0,
                nms_thresh=0.7,
                post_nms_top_n=1000,
                pre_nms_top_n=1000),
            anchor_start_size=32,
            num_chan=256,
            min_level=2,
            max_level=6)

    def roi_extractor(self):
        return self.faster_rcnn.FPNRoIAlign(
            canconical_level=4,
            canonical_size=224,
            max_level=5,
            min_level=2,
            box_resolution=7,
            sampling_ratio=2)

    def bbox_head(self, num_classes):
        return self.faster_rcnn.BBoxHead(
            head=self.faster_rcnn.TwoFCHead(mlp_dim=1024),
            nms=self.faster_rcnn.MultiClassNMS(
                keep_top_k=100, nms_threshold=0.5, score_threshold=0.05),
            num_classes=num_classes)

    def bbox_assigner(self, num_classes):
        return self.faster_rcnn.BBoxAssigner(
            batch_size_per_im=512,
            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            bg_thresh_hi=0.5,
            bg_thresh_lo=0.0,
            fg_fraction=0.25,
            fg_thresh=0.5,
            class_nums=num_classes)

    def object_detection(self,
                         paths=None,
                         images=None,
                         use_gpu=False,
                         batch_size=1,
                         output_dir='detection_result',
                         score_thresh=0.5,
                         visualization=True):
        """API of Object Detection.

        :param paths: the path of images.
        :type paths: list, each element is correspond to the path of an image.
        :param images: data of images, [N, H, W, C]
        :type images: numpy.ndarray
        :param use_gpu: whether to use gpu or not.
        :type use_gpu: bool
        :param batch_size: bathc size.
        :type batch_size: int
        :param output_dir: the directory to store the detection result.
        :type output_dir: str
        :param score_thresh: the threshold of detection confidence.
        :type score_thresh: float
        :param visualization: whether to draw box and save images.
        :type visualization: bool
        """
        all_images = []
        paths = paths if paths else []
        for yield_data in self.faster_rcnn.test_reader(paths, images):
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
            padding_image, padding_info, padding_shape = self.faster_rcnn.padding_minibatch(
                batch_data, coarsest_stride=32, use_padded_im_info=True)
            padding_image_tensor = PaddleTensor(padding_image.copy())
            padding_info_tensor = PaddleTensor(padding_info.copy())
            padding_shape_tensor = PaddleTensor(padding_shape.copy())
            feed_list = [
                padding_image_tensor, padding_info_tensor, padding_shape_tensor
            ]
            if use_gpu:
                data_out = self.gpu_predictor.run(feed_list)
            else:
                data_out = self.cpu_predictor.run(feed_list)

            output = self.faster_rcnn.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
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
            '--input_path',
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
