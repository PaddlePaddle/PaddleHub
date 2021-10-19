# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


class MultiClassNMS(object):
    # __op__ = fluid.layers.multiclass_nms
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 background_label=0):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label


class SmoothL1Loss(object):
    '''
    Smooth L1 loss
    Args:
        sigma (float): hyper param in smooth l1 loss
    '''

    def __init__(self, sigma=1.0):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma

    def __call__(self, x, y, inside_weight=None, outside_weight=None):
        return fluid.layers.smooth_l1(
            x,
            y,
            inside_weight=inside_weight,
            outside_weight=outside_weight,
            sigma=self.sigma)


class BoxCoder(object):
    def __init__(self,
                 prior_box_var=[0.1, 0.1, 0.2, 0.2],
                 code_type='decode_center_size',
                 box_normalized=False,
                 axis=1):
        super(BoxCoder, self).__init__()
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.axis = axis


class TwoFCHead(object):
    """
    RCNN head with two Fully Connected layers

    Args:
        mlp_dim (int): num of filters for the fc layers
    """

    def __init__(self, mlp_dim=1024):
        super(TwoFCHead, self).__init__()
        self.mlp_dim = mlp_dim

    def __call__(self, roi_feat):
        fan = roi_feat.shape[1] * roi_feat.shape[2] * roi_feat.shape[3]

        fc6 = fluid.layers.fc(
            input=roi_feat,
            size=self.mlp_dim,
            act='relu',
            name='fc6',
            param_attr=ParamAttr(name='fc6_w', initializer=Xavier(fan_out=fan)),
            bias_attr=ParamAttr(
                name='fc6_b', learning_rate=2., regularizer=L2Decay(0.)))
        head_feat = fluid.layers.fc(
            input=fc6,
            size=self.mlp_dim,
            act='relu',
            name='fc7',
            param_attr=ParamAttr(name='fc7_w', initializer=Xavier()),
            bias_attr=ParamAttr(
                name='fc7_b', learning_rate=2., regularizer=L2Decay(0.)))

        return head_feat


class BBoxHead(object):
    """
    RCNN bbox head

    Args:
        head (object): the head module instance, e.g., `ResNetC5`, `TwoFCHead`
        box_coder (object): `BoxCoder` instance
        nms (object): `MultiClassNMS` instance
        num_classes: number of output classes
    """
    __inject__ = ['head', 'box_coder', 'nms', 'bbox_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 head,
                 box_coder=BoxCoder(),
                 nms=MultiClassNMS(),
                 bbox_loss=SmoothL1Loss(),
                 num_classes=81):
        super(BBoxHead, self).__init__()
        self.head = head
        self.num_classes = num_classes
        self.box_coder = box_coder
        self.nms = nms
        self.bbox_loss = bbox_loss
        self.head_feat = None

    def get_head_feat(self, input=None):
        """
        Get the bbox head feature map.
        """

        if input is not None:
            feat = self.head(input)
            if isinstance(feat, OrderedDict):
                feat = list(feat.values())[0]
            self.head_feat = feat
        return self.head_feat

    def _get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.

        Returns:
            cls_score(Variable): Output of rpn head with shape of
                [N, num_anchors, H, W].
            bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        head_feat = self.get_head_feat(roi_feat)
        # when ResNetC5 output a single feature map
        if not isinstance(self.head, TwoFCHead):
            head_feat = fluid.layers.pool2d(
                head_feat, pool_type='avg', global_pooling=True)
        cls_score = fluid.layers.fc(
            input=head_feat,
            size=self.num_classes,
            act=None,
            name='cls_score',
            param_attr=ParamAttr(
                name='cls_score_w', initializer=Normal(loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='cls_score_b', learning_rate=2., regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(
            input=head_feat,
            size=4 * self.num_classes,
            act=None,
            name='bbox_pred',
            param_attr=ParamAttr(
                name='bbox_pred_w', initializer=Normal(loc=0.0, scale=0.001)),
            bias_attr=ParamAttr(
                name='bbox_pred_b', learning_rate=2., regularizer=L2Decay(0.)))
        return cls_score, bbox_pred

    def get_loss(self, roi_feat, labels_int32, bbox_targets,
                 bbox_inside_weights, bbox_outside_weights):
        """
        Get bbox_head loss.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            labels_int32(Variable): Class label of a RoI with shape [P, 1].
                P is the number of RoI.
            bbox_targets(Variable): Box label of a RoI with shape
                [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.

        Return:
            Type: Dict
                loss_cls(Variable): bbox_head loss.
                loss_bbox(Variable): bbox_head loss.
        """

        cls_score, bbox_pred = self._get_output(roi_feat)

        labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        loss_bbox = self.bbox_loss(
            x=bbox_pred,
            y=bbox_targets,
            inside_weight=bbox_inside_weights,
            outside_weight=bbox_outside_weights)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}

    def get_prediction(self,
                       roi_feat,
                       rois,
                       im_info,
                       im_shape,
                       return_box_score=False):
        """
        Get prediction bounding box in test stage.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            rois (Variable): Output of generate_proposals in rpn head.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.
            im_shape (Variable): Actual shape of original image with shape
                [B, 3]. B is the number of images, each element consists of
                original_height, original_width, 1

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
                N is the total number of prediction.
        """
        cls_score, bbox_pred = self._get_output(roi_feat)

        im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        im_scale = fluid.layers.sequence_expand(im_scale, rois)
        boxes = rois / im_scale
        cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(bbox_pred, (-1, self.num_classes, 4))
        # self.box_coder
        decoded_box = fluid.layers.box_coder(
            prior_box=boxes,
            target_box=bbox_pred,
            prior_box_var=self.box_coder.prior_box_var,
            code_type=self.box_coder.code_type,
            box_normalized=self.box_coder.box_normalized,
            axis=self.box_coder.axis)
        cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=im_shape)
        if return_box_score:
            return {'bbox': cliped_box, 'score': cls_prob}
        # self.nms
        pred_result = fluid.layers.multiclass_nms(
            bboxes=cliped_box,
            scores=cls_prob,
            score_threshold=self.nms.score_threshold,
            nms_top_k=self.nms.nms_top_k,
            keep_top_k=self.nms.keep_top_k,
            nms_threshold=self.nms.nms_threshold,
            normalized=self.nms.normalized,
            nms_eta=self.nms.nms_eta,
            background_label=self.nms.background_label)
        return pred_result
