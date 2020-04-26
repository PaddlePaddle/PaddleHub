# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.common.paddle_helper import add_vars_prefix

from ssd.data_feed import reader, DecodeImage, ResizeImage, NormalizeImage, Permute
from ssd.processor import load_label_info, postprocess
from ssd.multi_box_head import MultiBoxHead
from ssd.output_decoder import SSDOutputDecoder


@moduleinfo(
    name="ssd",
    version="1.0.0",
    type="cv/object_detection",
    summary=
    "SSD (Single Shot MultiBox Detector) is a object detection model, which trained with PASCAL VOC dataset.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class SSD(hub.Module):
    def _initialize(self):
        self.reader = reader
        self.load_label_info = load_label_info
        self.postprocess = postprocess
        self.MultiBoxHead = MultiBoxHead
        self.SSDOutputDecoder = SSDOutputDecoder
        self.DecodeImage = DecodeImage
        self.ResizeImage = ResizeImage
        self.NormalizeImage = NormalizeImage
        self.Permute = Permute

    def context(self,
                body_feats,
                multi_box_head,
                ssd_output_decoder,
                image,
                trainable=True,
                var_prefix='',
                get_prediction=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param body_feats: feature mps of backbone outputs
        :type body_feats: list
        :param multi_box_head: SSD head of MultiBoxHead.
        :type multi_box_head: <class 'MultiBoxHead' object>
        :param ssd_output_decoder: SSD output decoder
        :type ssd_output_decoder: <class 'SSDOutputDecoder' object>
        :param image: image tensor.
        :type image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param var_prefix: the prefix of variables in ssd
        :type var_prefix: str
        :param get_prediction: whether to get prediction,
            if True, outputs is bbox_out,
            if False, outputs is body_features.
        :type get_prediction: bool
        """
        context_prog = image.block.program
        with fluid.program_guard(context_prog):
            im_size = fluid.layers.data(
                name='im_size', shape=[2], dtype='int32')
            inputs = {
                'image': var_prefix + image.name,
                'im_size': var_prefix + im_size.name
            }
            if not get_prediction:
                outputs = {
                    'body_features':
                    [var_prefix + var.name for var in body_feats]
                }
            else:
                locs, confs, box, box_var = fluid.layers.multi_box_head(
                    inputs=body_feats,
                    image=image,
                    base_size=multi_box_head.base_size,
                    num_classes=multi_box_head.num_classes,
                    aspect_ratios=multi_box_head.aspect_ratios,
                    min_ratio=multi_box_head.min_ratio,
                    max_ratio=multi_box_head.max_ratio,
                    min_sizes=multi_box_head.min_sizes,
                    max_sizes=multi_box_head.max_sizes,
                    steps=multi_box_head.steps,
                    offset=multi_box_head.offset,
                    flip=multi_box_head.flip,
                    kernel_size=multi_box_head.kernel_size,
                    pad=multi_box_head.pad,
                    min_max_aspect_ratios_order=multi_box_head.
                    min_max_aspect_ratios_order)
                pred = fluid.layers.detection_output(
                    loc=locs,
                    scores=confs,
                    prior_box=box,
                    prior_box_var=box_var,
                    nms_threshold=ssd_output_decoder.nms_threshold,
                    nms_top_k=ssd_output_decoder.nms_top_k,
                    keep_top_k=ssd_output_decoder.keep_top_k,
                    score_threshold=ssd_output_decoder.score_threshold,
                    nms_eta=ssd_output_decoder.nms_eta,
                    background_label=ssd_output_decoder.background_label)
                outputs = {'bbox_out': var_prefix + pred.name}

            add_vars_prefix(context_prog, var_prefix)
            add_vars_prefix(fluid.default_startup_program(), var_prefix)
            inputs = {
                key: context_prog.global_block().vars[value]
                for key, value in inputs.items()
            }
            outputs = {
                key: [
                    context_prog.global_block().vars[varname]
                    for varname in value
                ]
                for key, value in outputs.items()
            }

            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable

            return inputs, outputs, context_prog
