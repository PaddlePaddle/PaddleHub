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
from paddlehub.module.module import moduleinfo, runnable, serving

from paddle.inference import Config
from paddle.inference import create_predictor

from paddlehub.io.parser import txt_parser
from paddlehub.common.paddle_helper import add_vars_prefix

from faster_rcnn_resnet50_coco2017.processor import load_label_info, postprocess, base64_to_cv2
from faster_rcnn_resnet50_coco2017.data_feed import test_reader, padding_minibatch
from faster_rcnn_resnet50_coco2017.resnet import ResNet, ResNetC5
from faster_rcnn_resnet50_coco2017.rpn_head import AnchorGenerator, RPNTargetAssign, GenerateProposals, RPNHead
from faster_rcnn_resnet50_coco2017.bbox_head import MultiClassNMS, BBoxHead, SmoothL1Loss
from faster_rcnn_resnet50_coco2017.bbox_assigner import BBoxAssigner
from faster_rcnn_resnet50_coco2017.roi_extractor import RoIAlign


@moduleinfo(
    name="faster_rcnn_resnet50_coco2017",
    version="1.1.1",
    type="cv/object_detection",
    summary="Baidu's Faster R-CNN model for object detection with backbone ResNet50, trained with dataset COCO2017",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNResNet50(hub.Module):
    def _initialize(self):
        # default pretrained model, Faster-RCNN with backbone ResNet50, shape of input tensor is [3, 800, 1333]
        self.default_pretrained_model_path = os.path.join(self.directory, "faster_rcnn_resnet50_model")
        self.label_names = load_label_info(os.path.join(self.directory, "label_file.txt"))
        self._set_config()

    def _get_device_id(self, places):
        try:
            places = os.environ[places]
            id = int(places)
        except:
            id = -1
        return id

    def _set_config(self):
        """
        predictor config setting
        """

        # create default cpu predictor
        cpu_config = Config(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_predictor(cpu_config)

        # create predictors using various types of devices

        # npu
        npu_id = self._get_device_id("FLAGS_selected_npus")
        if npu_id != -1:
            # use npu
            npu_config = Config(self.default_pretrained_model_path)
            npu_config.disable_glog_info()
            npu_config.enable_npu(device_id=npu_id)
            self.npu_predictor = create_predictor(npu_config)

        # gpu
        gpu_id = self._get_device_id("CUDA_VISIBLE_DEVICES")
        if gpu_id != -1:
            # use gpu
            gpu_config = Config(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=gpu_id)
            self.gpu_predictor = create_predictor(gpu_config)

        # xpu
        xpu_id = self._get_device_id("XPU_VISIBLE_DEVICES")
        if xpu_id != -1:
            # use xpu
            xpu_config = Config(self.default_pretrained_model_path)
            xpu_config.disable_glog_info()
            xpu_config.enable_xpu(100)
            self.xpu_predictor = create_predictor(xpu_config)

    def context(self, num_classes=81, trainable=True, pretrained=True, phase='train'):
        """
        Distill the Head Features, so as to perform transfer learning.

        Args:
            num_classes (int): number of categories
            trainable (bool): whether to set parameters trainable.
            pretrained (bool): whether to load default pretrained model.
            phase (str): optional choices are 'train' and 'predict'.

        Returns:
             inputs (dict): the input variables.
             outputs (dict): the output variables.
             context_prog (Program): the program to execute transfer learning.
        """
        context_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            with fluid.unique_name.guard():
                image = fluid.layers.data(name='image', shape=[-1, 3, -1, -1], dtype='float32')
                # backbone
                backbone = ResNet(norm_type='affine_channel', depth=50, feature_maps=4, freeze_at=2)
                body_feats = backbone(image)

                # var_prefix
                var_prefix = '@HUB_{}@'.format(self.name)
                im_info = fluid.layers.data(name='im_info', shape=[3], dtype='float32', lod_level=0)
                im_shape = fluid.layers.data(name='im_shape', shape=[3], dtype='float32', lod_level=0)
                body_feat_names = list(body_feats.keys())
                # rpn_head: RPNHead
                rpn_head = self.rpn_head()
                rois = rpn_head.get_proposals(body_feats, im_info, mode=phase)
                # train
                if phase == 'train':
                    gt_bbox = fluid.layers.data(name='gt_bbox', shape=[4], dtype='float32', lod_level=1)
                    is_crowd = fluid.layers.data(name='is_crowd', shape=[1], dtype='int32', lod_level=1)
                    gt_class = fluid.layers.data(name='gt_class', shape=[1], dtype='int32', lod_level=1)
                    rpn_loss = rpn_head.get_loss(im_info, gt_bbox, is_crowd)
                    # bbox_assigner: BBoxAssigner
                    bbox_assigner = self.bbox_assigner(num_classes)
                    outs = fluid.layers.generate_proposal_labels(rpn_rois=rois,
                                                                 gt_classes=gt_class,
                                                                 is_crowd=is_crowd,
                                                                 gt_boxes=gt_bbox,
                                                                 im_info=im_info,
                                                                 batch_size_per_im=bbox_assigner.batch_size_per_im,
                                                                 fg_fraction=bbox_assigner.fg_fraction,
                                                                 fg_thresh=bbox_assigner.fg_thresh,
                                                                 bg_thresh_hi=bbox_assigner.bg_thresh_hi,
                                                                 bg_thresh_lo=bbox_assigner.bg_thresh_lo,
                                                                 bbox_reg_weights=bbox_assigner.bbox_reg_weights,
                                                                 class_nums=bbox_assigner.class_nums,
                                                                 use_random=bbox_assigner.use_random)
                    rois = outs[0]

                body_feat = body_feats[body_feat_names[-1]]
                # roi_extractor: RoIAlign
                roi_extractor = self.roi_extractor()
                roi_feat = fluid.layers.roi_align(input=body_feat,
                                                  rois=rois,
                                                  pooled_height=roi_extractor.pooled_height,
                                                  pooled_width=roi_extractor.pooled_width,
                                                  spatial_scale=roi_extractor.spatial_scale,
                                                  sampling_ratio=roi_extractor.sampling_ratio)
                # head_feat
                bbox_head = self.bbox_head(num_classes)
                head_feat = bbox_head.head(roi_feat)
                if isinstance(head_feat, OrderedDict):
                    head_feat = list(head_feat.values())[0]
                if phase == 'train':
                    inputs = {
                        'image': var_prefix + image.name,
                        'im_info': var_prefix + im_info.name,
                        'im_shape': var_prefix + im_shape.name,
                        'gt_class': var_prefix + gt_class.name,
                        'gt_bbox': var_prefix + gt_bbox.name,
                        'is_crowd': var_prefix + is_crowd.name
                    }
                    outputs = {
                        'head_features': var_prefix + head_feat.name,
                        'rpn_cls_loss': var_prefix + rpn_loss['rpn_cls_loss'].name,
                        'rpn_reg_loss': var_prefix + rpn_loss['rpn_reg_loss'].name,
                        'generate_proposal_labels': [var_prefix + var.name for var in outs]
                    }
                elif phase == 'predict':
                    pred = bbox_head.get_prediction(roi_feat, rois, im_info, im_shape)
                    inputs = {
                        'image': var_prefix + image.name,
                        'im_info': var_prefix + im_info.name,
                        'im_shape': var_prefix + im_shape.name
                    }
                    outputs = {
                        'head_features': var_prefix + head_feat.name,
                        'rois': var_prefix + rois.name,
                        'bbox_out': var_prefix + pred.name
                    }
                add_vars_prefix(context_prog, var_prefix)
                add_vars_prefix(startup_program, var_prefix)

                global_vars = context_prog.global_block().vars
                inputs = {key: global_vars[value] for key, value in inputs.items()}
                outputs = {
                    key: global_vars[value] if not isinstance(value, list) else [global_vars[var] for var in value]
                    for key, value in outputs.items()
                }

                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                exe.run(startup_program)
                if pretrained:

                    def _if_exist(var):
                        if num_classes != 81:
                            if 'bbox_pred' in var.name or 'cls_score' in var.name:
                                return False
                        return os.path.exists(os.path.join(self.default_pretrained_model_path, var.name))

                    fluid.io.load_vars(exe, self.default_pretrained_model_path, predicate=_if_exist)
                return inputs, outputs, context_prog

    def rpn_head(self):
        return RPNHead(anchor_generator=AnchorGenerator(anchor_sizes=[32, 64, 128, 256, 512],
                                                        aspect_ratios=[0.5, 1.0, 2.0],
                                                        stride=[16.0, 16.0],
                                                        variance=[1.0, 1.0, 1.0, 1.0]),
                       rpn_target_assign=RPNTargetAssign(rpn_batch_size_per_im=256,
                                                         rpn_fg_fraction=0.5,
                                                         rpn_negative_overlap=0.3,
                                                         rpn_positive_overlap=0.7,
                                                         rpn_straddle_thresh=0.0),
                       train_proposal=GenerateProposals(min_size=0.0,
                                                        nms_thresh=0.7,
                                                        post_nms_top_n=12000,
                                                        pre_nms_top_n=2000),
                       test_proposal=GenerateProposals(min_size=0.0,
                                                       nms_thresh=0.7,
                                                       post_nms_top_n=6000,
                                                       pre_nms_top_n=1000))

    def roi_extractor(self):
        return RoIAlign(resolution=14, sampling_ratio=0, spatial_scale=0.0625)

    def bbox_head(self, num_classes):
        return BBoxHead(head=ResNetC5(depth=50, norm_type='affine_channel'),
                        nms=MultiClassNMS(keep_top_k=100, nms_threshold=0.5, score_threshold=0.05),
                        bbox_loss=SmoothL1Loss(),
                        num_classes=num_classes)

    def bbox_assigner(self, num_classes):
        return BBoxAssigner(batch_size_per_im=512,
                            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                            bg_thresh_hi=0.5,
                            bg_thresh_lo=0.0,
                            fg_fraction=0.25,
                            fg_thresh=0.5,
                            class_nums=num_classes)

    def save_inference_model(self, dirname, model_filename=None, params_filename=None, combined=True):
        if combined:
            model_filename = "__model__" if not model_filename else model_filename
            params_filename = "__params__" if not params_filename else params_filename
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feeded_var_names, target_vars = fluid.io.load_inference_model(
            dirname=self.default_pretrained_model_path, executor=exe)

        fluid.io.save_inference_model(dirname=dirname,
                                      main_program=program,
                                      executor=exe,
                                      feeded_var_names=feeded_var_names,
                                      target_vars=target_vars,
                                      model_filename=model_filename,
                                      params_filename=params_filename)

    def object_detection(self,
                         paths=None,
                         images=None,
                         data=None,
                         use_gpu=False,
                         batch_size=1,
                         output_dir='detection_result',
                         score_thresh=0.5,
                         visualization=True,
                         use_device=None):
        """API of Object Detection.

        Args:
            paths (list[str]): The paths of images.
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            score_thresh (float): threshold for object detecion.
            use_device (str): use cpu, gpu, xpu or npu, overwrites use_gpu flag.

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
        # real predictor to use
        if use_device is not None:
            if use_device == "cpu":
                predictor = self.cpu_predictor
            elif use_device == "xpu":
                predictor = self.xpu_predictor
            elif use_device == "npu":
                predictor = self.npu_predictor
            elif use_device == "gpu":
                predictor = self.gpu_predictor
            else:
                raise Exception("Unsupported device: " + use_device)
        else:
            # use_device is not set, therefore follow use_gpu
            if use_gpu:
                predictor = self.gpu_predictor
            else:
                predictor = self.cpu_predictor

        paths = paths if paths else list()
        if data and 'image' in data:
            paths += data['image']

        all_images = list()
        for yield_return in test_reader(paths, images):
            all_images.append(yield_return)

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

            padding_image, padding_info, padding_shape = padding_minibatch(batch_data)

            input_names = predictor.get_input_names()

            padding_image_tensor = predictor.get_input_handle(input_names[0])
            padding_image_tensor.reshape(padding_image.shape)
            padding_image_tensor.copy_from_cpu(padding_image.copy())

            padding_info_tensor = predictor.get_input_handle(input_names[1])
            padding_info_tensor.reshape(padding_info.shape)
            padding_info_tensor.copy_from_cpu(padding_info.copy())

            padding_shape_tensor = predictor.get_input_handle(input_names[2])
            padding_shape_tensor.reshape(padding_shape.shape)
            padding_shape_tensor.copy_from_cpu(padding_shape.copy())

            predictor.run()
            output_names = predictor.get_output_names()
            output_handle = predictor.get_output_handle(output_names[0])

            output = postprocess(paths=paths,
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
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use GPU or not")

        self.arg_config_group.add_argument('--batch_size', type=int, default=1, help="batch size for prediction")
        self.arg_config_group.add_argument('--use_device',
                                           choices=["cpu", "gpu", "xpu", "npu"],
                                           help="use cpu, gpu, xpu or npu. overwrites use_gpu flag.")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument('--input_path', type=str, default=None, help="input data")

        self.arg_input_group.add_argument('--input_file', type=str, default=None, help="file contain input data")

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
        self.parser = argparse.ArgumentParser(description="Run the {}".format(self.name),
                                              prog="hub run {}".format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
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
                    raise RuntimeError("File %s or %s is not exist." % image_path)
        return self.object_detection(paths=input_data,
                                     use_gpu=args.use_gpu,
                                     batch_size=args.batch_size,
                                     use_device=args.use_device)
