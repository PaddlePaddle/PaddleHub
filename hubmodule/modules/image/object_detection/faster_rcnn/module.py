# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.common.paddle_helper import add_vars_prefix

from faster_rcnn.data_feed import test_reader, padding_minibatch
from faster_rcnn.processor import load_label_info, postprocess
from faster_rcnn.bbox_head import MultiClassNMS, BBoxHead, SmoothL1Loss, TwoFCHead
from faster_rcnn.rpn_head import AnchorGenerator, RPNTargetAssign, GenerateProposals, RPNHead, FPNRPNHead
from faster_rcnn.bbox_assigner import BBoxAssigner
from faster_rcnn.roi_extractor import RoIAlign, FPNRoIAlign


@moduleinfo(
    name="faster_rcnn",
    version="1.0.0",
    type="cv/object_detection",
    summary="Baidu's Faster R-CNN model for object detection.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com")
class FasterRCNNBase(hub.Module):
    def _initialize(self):
        # data_feed
        self.test_reader = test_reader
        self.padding_minibatch = padding_minibatch
        # processor
        self.load_label_info = load_label_info
        self.postprocess = postprocess
        # bbox_head
        self.MultiClassNMS = MultiClassNMS
        self.TwoFCHead = TwoFCHead
        self.BBoxHead = BBoxHead
        self.SmoothL1Loss = SmoothL1Loss
        # rpn_head
        self.AnchorGenerator = AnchorGenerator
        self.RPNTargetAssign = RPNTargetAssign
        self.GenerateProposals = GenerateProposals
        self.RPNHead = RPNHead
        self.FPNRPNHead = FPNRPNHead
        # bbox_assigner
        self.BBoxAssigner = BBoxAssigner
        # roi_extractor
        self.RoIAlign = RoIAlign
        self.FPNRoIAlign = FPNRoIAlign

    def context(self, body_feats, fpn, rpn_head, roi_extractor, bbox_head,
                bbox_assigner, image, trainable, var_prefix, phase):
        """Distill the Head Features, so as to perform transfer learning.

        :param body_feats: feature map of image classification to distill feature map.
        :type body_feats: list
        :param fpn: Feature Pyramid Network.
        :type fpn: <class 'FPN' object>
        :param rpn_head: Head of Region Proposal Network.
        :type rpn_head: <class 'RPNHead' object> or  <class 'FPNRPNHead' object>
        :param roi_extractor:
        :type roi_extractor:
        :param bbox_head: Head of Bounding Box.
        :type bbox_head: <class 'BBoxHead' object>
        :param bbox_assigner: Parameters of fluid.layers.generate_proposal_labels.
        :type bbox_assigner: <class 'BBoxAssigner' object>
        :param image: image tensor.
        :type image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param var_prefix: the prefix of variables in faster_rcnn
        :type var_prefix: str
        :param phase: Optional Choice: 'predict', 'train'
        :type phase: str
        """
        context_prog = image.block.program
        with fluid.program_guard(context_prog):
            im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32', lod_level=0)
            im_shape = fluid.layers.data(
                name='im_shape', shape=[3], dtype='float32', lod_level=0)
            #body_feats = backbone(image)
            body_feat_names = list(body_feats.keys())
            # fpn
            if fpn is not None:
                body_feats, spatial_scale = fpn.get_output(body_feats)
            # rpn_head: RPNHead
            rois = rpn_head.get_proposals(body_feats, im_info, mode=phase)
            # train
            if phase == 'train':
                gt_bbox = fluid.layers.data(
                    name='gt_bbox', shape=[4], dtype='float32', lod_level=1)
                is_crowd = fluid.layers.data(
                    name='is_crowd', shape=[1], dtype='int32', lod_level=1)
                gt_class = fluid.layers.data(
                    name='gt_class', shape=[1], dtype='int32', lod_level=1)
                rpn_loss = rpn_head.get_loss(im_info, gt_bbox, is_crowd)
                # bbox_assigner: BBoxAssigner
                outs = fluid.layers.generate_proposal_labels(
                    rpn_rois=rois,
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
            if fpn is None:
                body_feat = body_feats[body_feat_names[-1]]
                # roi_extractor: RoIAlign
                roi_feat = fluid.layers.roi_align(
                    input=body_feat,
                    rois=rois,
                    pooled_height=roi_extractor.pooled_height,
                    pooled_width=roi_extractor.pooled_width,
                    spatial_scale=roi_extractor.spatial_scale,
                    sampling_ratio=roi_extractor.sampling_ratio)
            else:
                # roi_extractor: FPNRoIAlign
                roi_feat = roi_extractor(
                    head_inputs=body_feats,
                    rois=rois,
                    spatial_scale=spatial_scale)
            # head_feat
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
                    'head_feat':
                    var_prefix + head_feat.name,
                    'rpn_cls_loss':
                    var_prefix + rpn_loss['rpn_cls_loss'].name,
                    'rpn_reg_loss':
                    var_prefix + rpn_loss['rpn_reg_loss'].name,
                    'generate_proposal_labels':
                    [var_prefix + var.name for var in outs]
                }
            elif phase == 'predict':
                pred = bbox_head.get_prediction(roi_feat, rois, im_info,
                                                im_shape)
                inputs = {
                    'image': var_prefix + image.name,
                    'im_info': var_prefix + im_info.name,
                    'im_shape': var_prefix + im_shape.name
                }
                outputs = {
                    'head_feat': var_prefix + head_feat.name,
                    'rois': var_prefix + rois.name,
                    'bbox_out': var_prefix + pred.name
                }
            add_vars_prefix(context_prog, var_prefix)
            add_vars_prefix(fluid.default_startup_program(), var_prefix)

            global_vars = context_prog.global_block().vars
            inputs = {key: global_vars[value] for key, value in inputs.items()}
            outputs = {
                key: global_vars[value] if not isinstance(value, list) else
                [global_vars[var] for var in value]
                for key, value in outputs.items()
            }
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            for param in context_prog.global_block().iter_parameters():
                param.trainable = trainable
            return inputs, outputs, context_prog
