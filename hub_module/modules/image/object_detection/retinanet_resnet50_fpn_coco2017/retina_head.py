# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.regularizer import L2Decay

__all__ = [
    'AnchorGenerator', 'RetinaTargetAssign', 'RetinaOutputDecoder', 'RetinaHead'
]


class AnchorGenerator(object):
    # __op__ = fluid.layers.anchor_generator
    def __init__(self,
                 stride=[16.0, 16.0],
                 anchor_sizes=[32, 64, 128, 256, 512],
                 aspect_ratios=[0.5, 1., 2.],
                 variance=[1., 1., 1., 1.]):
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios
        self.variance = variance
        self.stride = stride


class RetinaTargetAssign(object):
    # __op__ = fluid.layers.retinanet_target_assign
    def __init__(self, positive_overlap=0.5, negative_overlap=0.4):
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap


class RetinaOutputDecoder(object):
    # __op__ = fluid.layers.retinanet_detection_output
    def __init__(self,
                 score_thresh=0.05,
                 nms_thresh=0.3,
                 pre_nms_top_n=1000,
                 detections_per_im=100,
                 nms_eta=1.0):
        super(RetinaOutputDecoder, self).__init__()
        self.score_threshold = score_thresh
        self.nms_threshold = nms_thresh
        self.nms_top_k = pre_nms_top_n
        self.keep_top_k = detections_per_im
        self.nms_eta = nms_eta


class RetinaHead(object):
    """
    Retina Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        target_assign (object): `RetinaTargetAssign` instance
        output_decoder (object): `RetinaOutputDecoder` instance
        num_convs_per_octave (int): Number of convolution layers in each octave
        num_chan (int): Number of octave output channels
        max_level (int): Highest level of FPN output
        min_level (int): Lowest level of FPN output
        prior_prob (float): Used to set the bias init for the class prediction layer
        base_scale (int): Anchors are generated based on this scale
        num_scales_per_octave (int): Number of anchor scales per octave
        num_classes (int): Number of classes
        gamma (float): The parameter in focal loss
        alpha (float): The parameter in focal loss
        sigma (float): The parameter in smooth l1 loss
    """
    __inject__ = ['anchor_generator', 'target_assign', 'output_decoder']
    __shared__ = ['num_classes']

    def __init__(self,
                 anchor_generator=AnchorGenerator(),
                 target_assign=RetinaTargetAssign(),
                 output_decoder=RetinaOutputDecoder(),
                 num_convs_per_octave=4,
                 num_chan=256,
                 max_level=7,
                 min_level=3,
                 prior_prob=0.01,
                 base_scale=4,
                 num_scales_per_octave=3,
                 num_classes=81,
                 gamma=2.0,
                 alpha=0.25,
                 sigma=3.0151134457776365):
        self.anchor_generator = anchor_generator
        self.target_assign = target_assign
        self.output_decoder = output_decoder
        self.num_convs_per_octave = num_convs_per_octave
        self.num_chan = num_chan
        self.max_level = max_level
        self.min_level = min_level
        self.prior_prob = prior_prob
        self.base_scale = base_scale
        self.num_scales_per_octave = num_scales_per_octave
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

    def _class_subnet(self, body_feats, spatial_scale):
        """
        Get class predictions of all level FPN level.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.

        Returns:
            cls_pred_input(list): Class prediction of all input fpn levels.
        """
        assert len(body_feats) == self.max_level - self.min_level + 1
        fpn_name_list = list(body_feats.keys())
        cls_pred_list = []
        for lvl in range(self.min_level, self.max_level + 1):
            fpn_name = fpn_name_list[self.max_level - lvl]
            subnet_blob = body_feats[fpn_name]
            for i in range(self.num_convs_per_octave):
                conv_name = 'retnet_cls_conv_n{}_fpn{}'.format(i, lvl)
                conv_share_name = 'retnet_cls_conv_n{}_fpn{}'.format(
                    i, self.min_level)
                subnet_blob_in = subnet_blob
                subnet_blob = fluid.layers.conv2d(
                    input=subnet_blob_in,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act='relu',
                    name=conv_name,
                    param_attr=ParamAttr(
                        name=conv_share_name + '_w',
                        initializer=Normal(loc=0., scale=0.01)),
                    bias_attr=ParamAttr(
                        name=conv_share_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))

            # class prediction
            cls_name = 'retnet_cls_pred_fpn{}'.format(lvl)
            cls_share_name = 'retnet_cls_pred_fpn{}'.format(self.min_level)
            num_anchors = self.num_scales_per_octave * len(
                self.anchor_generator.aspect_ratios)
            cls_dim = num_anchors * (self.num_classes - 1)
            # bias initialization: b = -log((1 - pai) / pai)
            bias_init = float(-np.log((1 - self.prior_prob) / self.prior_prob))
            out_cls = fluid.layers.conv2d(
                input=subnet_blob,
                num_filters=cls_dim,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=cls_name,
                param_attr=ParamAttr(
                    name=cls_share_name + '_w',
                    initializer=Normal(loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=cls_share_name + '_b',
                    initializer=Constant(value=bias_init),
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            cls_pred_list.append(out_cls)

        return cls_pred_list

    def _bbox_subnet(self, body_feats, spatial_scale):
        """
        Get bounding box predictions of all level FPN level.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.

        Returns:
            bbox_pred_input(list): Bounding box prediction of all input fpn
                levels.
        """
        assert len(body_feats) == self.max_level - self.min_level + 1
        fpn_name_list = list(body_feats.keys())
        bbox_pred_list = []
        for lvl in range(self.min_level, self.max_level + 1):
            fpn_name = fpn_name_list[self.max_level - lvl]
            subnet_blob = body_feats[fpn_name]
            for i in range(self.num_convs_per_octave):
                conv_name = 'retnet_bbox_conv_n{}_fpn{}'.format(i, lvl)
                conv_share_name = 'retnet_bbox_conv_n{}_fpn{}'.format(
                    i, self.min_level)
                subnet_blob_in = subnet_blob
                subnet_blob = fluid.layers.conv2d(
                    input=subnet_blob_in,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=1,
                    padding=1,
                    act='relu',
                    name=conv_name,
                    param_attr=ParamAttr(
                        name=conv_share_name + '_w',
                        initializer=Normal(loc=0., scale=0.01)),
                    bias_attr=ParamAttr(
                        name=conv_share_name + '_b',
                        learning_rate=2.,
                        regularizer=L2Decay(0.)))

            # bbox prediction
            bbox_name = 'retnet_bbox_pred_fpn{}'.format(lvl)
            bbox_share_name = 'retnet_bbox_pred_fpn{}'.format(self.min_level)
            num_anchors = self.num_scales_per_octave * len(
                self.anchor_generator.aspect_ratios)
            bbox_dim = num_anchors * 4
            out_bbox = fluid.layers.conv2d(
                input=subnet_blob,
                num_filters=bbox_dim,
                filter_size=3,
                stride=1,
                padding=1,
                act=None,
                name=bbox_name,
                param_attr=ParamAttr(
                    name=bbox_share_name + '_w',
                    initializer=Normal(loc=0., scale=0.01)),
                bias_attr=ParamAttr(
                    name=bbox_share_name + '_b',
                    learning_rate=2.,
                    regularizer=L2Decay(0.)))
            bbox_pred_list.append(out_bbox)
        return bbox_pred_list

    def _anchor_generate(self, body_feats, spatial_scale):
        """
        Get anchor boxes of all level FPN level.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.

        Return:
            anchor_input(list): Anchors of all input fpn levels with shape of.
            anchor_var_input(list): Anchor variance of all input fpn levels with shape.
        """
        assert len(body_feats) == self.max_level - self.min_level + 1
        fpn_name_list = list(body_feats.keys())
        anchor_list = []
        anchor_var_list = []
        for lvl in range(self.min_level, self.max_level + 1):
            anchor_sizes = []
            stride = int(1 / spatial_scale[self.max_level - lvl])
            for octave in range(self.num_scales_per_octave):
                anchor_size = stride * (2**(float(octave) / float(
                    self.num_scales_per_octave))) * self.base_scale
                anchor_sizes.append(anchor_size)
            fpn_name = fpn_name_list[self.max_level - lvl]
            anchor, anchor_var = fluid.layers.anchor_generator(
                input=body_feats[fpn_name],
                anchor_sizes=anchor_sizes,
                aspect_ratios=self.anchor_generator.aspect_ratios,
                stride=[stride, stride],
                variance=self.anchor_generator.variance)
            anchor_list.append(anchor)
            anchor_var_list.append(anchor_var)
        return anchor_list, anchor_var_list

    def _get_output(self, body_feats, spatial_scale):
        """
        Get class, bounding box predictions and anchor boxes of all level FPN level.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.

        Returns:
            cls_pred_input(list): Class prediction of all input fpn levels.
            bbox_pred_input(list): Bounding box prediction of all input fpn
                levels.
            anchor_input(list): Anchors of all input fpn levels with shape of.
            anchor_var_input(list): Anchor variance of all input fpn levels with
                shape.
        """
        assert len(body_feats) == self.max_level - self.min_level + 1
        # class subnet
        cls_pred_list = self._class_subnet(body_feats, spatial_scale)
        # bbox subnet
        bbox_pred_list = self._bbox_subnet(body_feats, spatial_scale)
        #generate anchors
        anchor_list, anchor_var_list = self._anchor_generate(
            body_feats, spatial_scale)
        cls_pred_reshape_list = []
        bbox_pred_reshape_list = []
        anchor_reshape_list = []
        anchor_var_reshape_list = []
        for i in range(self.max_level - self.min_level + 1):
            cls_pred_transpose = fluid.layers.transpose(
                cls_pred_list[i], perm=[0, 2, 3, 1])
            cls_pred_reshape = fluid.layers.reshape(
                cls_pred_transpose, shape=(0, -1, self.num_classes - 1))
            bbox_pred_transpose = fluid.layers.transpose(
                bbox_pred_list[i], perm=[0, 2, 3, 1])
            bbox_pred_reshape = fluid.layers.reshape(
                bbox_pred_transpose, shape=(0, -1, 4))
            anchor_reshape = fluid.layers.reshape(anchor_list[i], shape=(-1, 4))
            anchor_var_reshape = fluid.layers.reshape(
                anchor_var_list[i], shape=(-1, 4))
            cls_pred_reshape_list.append(cls_pred_reshape)
            bbox_pred_reshape_list.append(bbox_pred_reshape)
            anchor_reshape_list.append(anchor_reshape)
            anchor_var_reshape_list.append(anchor_var_reshape)
        output = {}
        output['cls_pred'] = cls_pred_reshape_list
        output['bbox_pred'] = bbox_pred_reshape_list
        output['anchor'] = anchor_reshape_list
        output['anchor_var'] = anchor_var_reshape_list
        return output

    def get_prediction(self, body_feats, spatial_scale, im_info):
        """
        Get prediction bounding box in test stage.

        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
                N is the total number of prediction.
        """
        output = self._get_output(body_feats, spatial_scale)
        cls_pred_reshape_list = output['cls_pred']
        bbox_pred_reshape_list = output['bbox_pred']
        anchor_reshape_list = output['anchor']
        for i in range(self.max_level - self.min_level + 1):
            cls_pred_reshape_list[i] = fluid.layers.sigmoid(
                cls_pred_reshape_list[i])
        pred_result = fluid.layers.retinanet_detection_output(
            bboxes=bbox_pred_reshape_list,
            scores=cls_pred_reshape_list,
            anchors=anchor_reshape_list,
            im_info=im_info,
            score_threshold=self.output_decoder.score_threshold,
            nms_threshold=self.output_decoder.nms_threshold,
            nms_top_k=self.output_decoder.nms_top_k,
            keep_top_k=self.output_decoder.keep_top_k,
            nms_eta=self.output_decoder.nms_eta)
        return pred_result

    def get_loss(self, body_feats, spatial_scale, im_info, gt_box, gt_label,
                 is_crowd):
        """
        Calculate the loss of retinanet.
        Args:
            fpn_dict(dict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
            im_info(Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.
            gt_box(Variable): The ground-truth bounding boxes with shape [M, 4].
                M is the number of groundtruth.
            gt_label(Variable): The ground-truth labels with shape [M, 1].
                M is the number of groundtruth.
            is_crowd(Variable): Indicates groud-truth is crowd or not with
                shape [M, 1]. M is the number of groundtruth.

        Returns:
            Type: dict
                loss_cls(Variable): focal loss.
                loss_bbox(Variable): smooth l1 loss.
        """
        output = self._get_output(body_feats, spatial_scale)
        cls_pred_reshape_list = output['cls_pred']
        bbox_pred_reshape_list = output['bbox_pred']
        anchor_reshape_list = output['anchor']
        anchor_var_reshape_list = output['anchor_var']

        cls_pred_input = fluid.layers.concat(cls_pred_reshape_list, axis=1)
        bbox_pred_input = fluid.layers.concat(bbox_pred_reshape_list, axis=1)
        anchor_input = fluid.layers.concat(anchor_reshape_list, axis=0)
        anchor_var_input = fluid.layers.concat(anchor_var_reshape_list, axis=0)
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight, fg_num = \
            fluid.layers.rpn_target_assign(
                bbox_pred=bbox_pred_input,
                cls_logits=cls_pred_input,
                anchor_box=anchor_input,
                anchor_var=anchor_var_input,
                gt_boxes=gt_box,
                gt_labels=gt_label,
                is_crowd=is_crowd,
                im_info=im_info,
                num_classes=self.num_classes - 1,
                rpn_batch_size_per_im=self.target_assign.rpn_batch_size_per_im,
                rpn_straddle_thresh=self.target_assign.rpn_straddle_thresh,
                rpn_fg_fraction=self.target_assign.rpn_fg_fraction,
                rpn_positive_overlap=self.target_assign.rpn_positive_overlap,
                rpn_negative_overlap=self.target_assign.rpn_negative_overlap,
                use_random=self.target_assign.use_random)
        fg_num = fluid.layers.reduce_sum(fg_num, name='fg_num')
        score_tgt = fluid.layers.cast(score_tgt, 'int32')
        loss_cls = fluid.layers.sigmoid_focal_loss(
            x=score_pred,
            label=score_tgt,
            fg_num=fg_num,
            gamma=self.gamma,
            alpha=self.alpha)
        loss_cls = fluid.layers.reduce_sum(loss_cls, name='loss_cls')
        loss_bbox = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=self.sigma,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        loss_bbox = fluid.layers.reduce_sum(loss_bbox, name='loss_bbox')
        loss_bbox = loss_bbox / fg_num
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}
