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
import yaml

from ssd_vgg16_512_coco2017.vgg import VGG


@moduleinfo(
    name="ssd_vgg16_512_coco2017",
    version="1.0.0",
    type="cv/object_detection",
    summary="SSD with backbone VGG16, trained with dataset COCO.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class SSDVGG16(hub.Module):
    def _initialize(self):
        self.ssd = hub.Module(name="ssd")
        # default pretrained model of SSD, the shape of input image tensor is (3, 512, 512)
        self.default_pretrained_model_path = os.path.join(
            self.directory, "ssd_vgg16_512_model")
        self.label_names = self.ssd.load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.image = None
        self.bbox_out = None
        self._set_config()
        self._config = None

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
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
        wrapped_prog = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(wrapped_prog, startup_program):
            with fluid.unique_name.guard():
                # image
                image = fluid.layers.data(
                    name='image', shape=[3, 512, 512], dtype='float32')
                # backbone
                backbone = VGG(
                    depth=16,
                    with_extra_blocks=True,
                    normalizations=[20., -1, -1, -1, -1, -1, -1],
                    extra_block_filters=[[256, 512, 1, 2,
                                          3], [128, 256, 1, 2, 3],
                                         [128, 256, 1, 2,
                                          3], [128, 256, 1, 2, 3],
                                         [128, 256, 1, 1, 4]])
                body_feats = backbone(image)
                # call ssd.context
                inputs, outputs, context_prog = self.ssd.context(
                    body_feats=body_feats,
                    multi_box_head=self.ssd.MultiBoxHead(
                        num_classes=num_classes, **self.multi_box_head_config),
                    ssd_output_decoder=self.ssd.SSDOutputDecoder(
                        **self.output_decoder_config),
                    image=image,
                    trainable=trainable,
                    var_prefix='@HUB_{}@'.format(self.name),
                    get_prediction=get_prediction)

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

    @property
    def config(self):
        if not self._config:
            with open(os.path.join(self.directory, 'config.yml')) as file:
                self._config = yaml.load(file.read(), Loader=yaml.FullLoader)
        return self._config

    @property
    def multi_box_head_config(self):
        return self.config['MultiBoxHead']

    @property
    def output_decoder_config(self):
        return self.config['SSDOutputDecoder']

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
        resize_image = self.ssd.ResizeImage(
            target_size=300, interp=1, max_size=0, use_cv2=False)
        data_reader = partial(
            self.ssd.reader, paths, images, resize_image=resize_image)
        batch_reader = fluid.io.batch(data_reader, batch_size=batch_size)
        paths = paths if paths else []
        res = []
        for iter_id, feed_data in enumerate(batch_reader()):
            np_data = np.array(feed_data).astype('float32')
            if np_data.shape == 1:
                np_data = np_data[0]
            else:
                np_data = np.squeeze(np_data, axis=1)
            data_tensor = PaddleTensor(np_data.copy())
            if use_gpu:
                data_out = self.gpu_predictor.run([data_tensor])
            else:
                data_out = self.cpu_predictor.run([data_tensor])
            output = self.ssd.postprocess(
                paths=paths,
                images=images,
                data_out=data_out,
                score_thresh=score_thresh,
                label_names=self.label_names,
                output_dir=output_dir,
                handle_id=iter_id * batch_size,
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
