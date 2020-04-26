# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ast
import argparse
from functools import partial

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.io.parser import txt_parser

from retinanet_resnet50_fpn_coco2017.fpn import FPN
from retinanet_resnet50_fpn_coco2017.retina_head import AnchorGenerator, RetinaTargetAssign, RetinaOutputDecoder, RetinaHead
from retinanet_resnet50_fpn_coco2017.processor import load_label_info, postprocess
from retinanet_resnet50_fpn_coco2017.data_feed import test_reader, padding_minibatch
from retinanet_resnet50_fpn_coco2017.resnet import ResNet


@moduleinfo(
    name="retinanet_resnet50_fpn_coco2017",
    version="1.0.0",
    type="cv/object_detection",
    summary=
    "Baidu's RetinaNet model for object detection, with backbone ResNet50 and FPN.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class RetinaNetResNet50FPN(hub.Module):
    def _initialize(self):
        # default pretrained model of Retinanet_ResNet50_FPN, the shape of input image tensor is (3, 608, 608)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "retinanet_resnet50_fpn_model")
        self.label_names = load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.im_info = None
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
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param get_prediction: whether to get prediction,
            if True, outputs is {'bbox_out': bbox_out},
            if False, outputs is {'head_features': head_features}.
        :type get_prediction: bool
        """
        context_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(context_prog, startup_program):
            # image
            image = fluid.layers.data(
                name='image',
                shape=[3, 800, 1333],
                dtype='float32',
                lod_level=0)
            # im_info
            im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32', lod_level=0)
            # backbone
            backbone = ResNet(
                norm_type='affine_channel',
                freeze_at=2,
                norm_decay=0.,
                depth=50,
                feature_maps=[3, 4, 5])
            body_feats = backbone(image)
            # retina_head
            retina_head = RetinaHead(
                anchor_generator=AnchorGenerator(
                    aspect_ratios=[1.0, 2.0, 0.5],
                    variance=[1.0, 1.0, 1.0, 1.0]),
                target_assign=RetinaTargetAssign(
                    positive_overlap=0.5, negative_overlap=0.4),
                output_decoder=RetinaOutputDecoder(
                    score_thresh=0.05,
                    nms_thresh=0.5,
                    pre_nms_top_n=1000,
                    detections_per_im=100,
                    nms_eta=1.0),
                num_convs_per_octave=4,
                num_chan=256,
                max_level=7,
                min_level=3,
                prior_prob=0.01,
                base_scale=4,
                num_scales_per_octave=3)
            # fpn
            fpn = FPN(
                max_level=7,
                min_level=3,
                num_chan=256,
                spatial_scale=[0.03125, 0.0625, 0.125],
                has_extra_convs=True)
            # body_feats
            body_feats, spatial_scale = fpn.get_output(body_feats)
            # inputs, outputs, context_prog
            inputs = {'image': image, 'im_info': im_info}
            if get_prediction:
                pred = retina_head.get_prediction(body_feats, spatial_scale,
                                                  im_info)
                outputs = {'bbox_out': pred}
            else:
                outputs = {'body_feats': body_feats}

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable
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
        :param visualization: whether to draw bounding box and save images.
        :type visualization: bool
        """
        all_images = []
        paths = paths if paths else []
        for yield_data in test_reader(paths, images):
            all_images.append(yield_data)

        images_num = len(all_images)
        loop_num = int(np.ceil(images_num / batch_size))
        res = []
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass
            padding_image, padding_info = padding_minibatch(
                batch_data, coarsest_stride=32, use_padded_im_info=True)
            padding_image_tensor = PaddleTensor(padding_image.copy())
            padding_info_tensor = PaddleTensor(padding_info.copy())
            feed_list = [padding_image_tensor, padding_info_tensor]
            if use_gpu:
                data_out = self.gpu_predictor.run(feed_list)
            else:
                data_out = self.cpu_predictor.run(feed_list)
            output = postprocess(
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
