import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant
from paddle.regularizer import L2Decay
from pycocotools.coco import COCO
from paddlehub.module.cv_module import Yolov3Module
from paddlehub.process.transforms import DetectTrainReader, DetectTestReader
from paddlehub.module.module import moduleinfo


class ConvBNLayer(nn.Layer):
    """Basic block for Darknet"""
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 filter_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 padding: int = 0,
                 act: str = 'leakly',
                 is_test: bool = False):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(ch_in,
                              ch_out,
                              filter_size,
                              padding=padding,
                              stride=stride,
                              groups=groups,
                              weight_attr=paddle.ParamAttr(initializer=Normal(0., 0.02)),
                              bias_attr=False)

        self.batch_norm = nn.BatchNorm(num_channels=ch_out,
                                       is_test=is_test,
                                       param_attr=paddle.ParamAttr(initializer=Normal(0., 0.02),
                                                                   regularizer=L2Decay(0.)))
        self.act = act

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv(inputs)
        out = self.batch_norm(out)
        if self.act == "leakly":
            out = F.leaky_relu(x=out, negative_slope=0.1)
        return out


class DownSample(nn.Layer):
    """Downsample block for Darknet"""
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 filter_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 is_test: bool = False):
        super(DownSample, self).__init__()

        self.conv_bn_layer = ConvBNLayer(ch_in=ch_in,
                                         ch_out=ch_out,
                                         filter_size=filter_size,
                                         stride=stride,
                                         padding=padding,
                                         is_test=is_test)
        self.ch_out = ch_out

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv_bn_layer(inputs)
        return out


class BasicBlock(nn.Layer):
    """Basic residual block for Darknet"""
    def __init__(self, ch_in: int, ch_out: int, is_test: bool = False):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, filter_size=1, stride=1, padding=0, is_test=is_test)
        self.conv2 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, filter_size=3, stride=1, padding=1, is_test=is_test)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        out = paddle.elementwise_add(x=inputs, y=conv2, act=None)
        return out


class LayerWarp(nn.Layer):
    """Warp layer composed by basic residual blocks"""
    def __init__(self, ch_in: int, ch_out: int, count: int, is_test: bool = False):
        super(LayerWarp, self).__init__()
        self.basicblock0 = BasicBlock(ch_in, ch_out, is_test=is_test)
        self.res_out_list = []
        for i in range(1, count):
            res_out = self.add_sublayer("basic_block_%d" % (i), BasicBlock(ch_out * 2, ch_out, is_test=is_test))
            self.res_out_list.append(res_out)
        self.ch_out = ch_out

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        y = self.basicblock0(inputs)
        for basic_block_i in self.res_out_list:
            y = basic_block_i(y)
        return y


DarkNet_cfg = {53: ([1, 2, 8, 8, 4])}


class DarkNet53_conv_body(nn.Layer):
    """Darknet53
    Args:
        ch_in(int): Input channels, default is 3.
        is_test (bool): Set the test mode, default is True.
    """
    def __init__(self, ch_in: int = 3, is_test: bool = False):
        super(DarkNet53_conv_body, self).__init__()
        self.stages = DarkNet_cfg[53]
        self.stages = self.stages[0:5]

        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=32, filter_size=3, stride=1, padding=1, is_test=is_test)

        self.downsample0 = DownSample(ch_in=32, ch_out=32 * 2, is_test=is_test)
        self.darknet53_conv_block_list = []
        self.downsample_list = []
        ch_in = [64, 128, 256, 512, 1024]

        for i, stage in enumerate(self.stages):
            conv_block = self.add_sublayer("stage_%d" % (i),
                                           LayerWarp(int(ch_in[i]), 32 * (2**i), stage, is_test=is_test))
            self.darknet53_conv_block_list.append(conv_block)

        for i in range(len(self.stages) - 1):
            downsample = self.add_sublayer(
                "stage_%d_downsample" % i, DownSample(ch_in=32 * (2**(i + 1)),
                                                      ch_out=32 * (2**(i + 2)),
                                                      is_test=is_test))
            self.downsample_list.append(downsample)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        out = self.conv0(inputs)
        out = self.downsample0(out)
        blocks = []
        for i, conv_block_i in enumerate(self.darknet53_conv_block_list):
            out = conv_block_i(out)
            blocks.append(out)
            if i < len(self.stages) - 1:
                out = self.downsample_list[i](out)
        return blocks[-1:-4:-1]


class YoloDetectionBlock(nn.Layer):
    """Basic block for Yolov3"""
    def __init__(self, ch_in: int, channel: int, is_test: bool = True):
        super(YoloDetectionBlock, self).__init__()

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)

        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=channel, filter_size=1, stride=1, padding=0, is_test=is_test)
        self.conv1 = ConvBNLayer(ch_in=channel, ch_out=channel * 2, filter_size=3, stride=1, padding=1, is_test=is_test)
        self.conv2 = ConvBNLayer(ch_in=channel * 2, ch_out=channel, filter_size=1, stride=1, padding=0, is_test=is_test)
        self.conv3 = ConvBNLayer(ch_in=channel, ch_out=channel * 2, filter_size=3, stride=1, padding=1, is_test=is_test)
        self.route = ConvBNLayer(ch_in=channel * 2, ch_out=channel, filter_size=1, stride=1, padding=0, is_test=is_test)
        self.tip = ConvBNLayer(ch_in=channel, ch_out=channel * 2, filter_size=3, stride=1, padding=1, is_test=is_test)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class Upsample(nn.Layer):
    """Upsample block for Yolov3"""
    def __init__(self, scale: int = 2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs: paddle.Tensor):
        shape_nchw = paddle.to_tensor(inputs.shape)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True
        out = F.resize_nearest(input=inputs, scale=self.scale, actual_shape=out_shape)
        return out


@moduleinfo(name="yolov3_darknet53_pascalvoc",
            type="CV/image_editing",
            author="paddlepaddle",
            author_email="",
            summary="Yolov3 is a detection model, this module is trained with VOC dataset.",
            version="1.0.0",
            meta=Yolov3Module)
class YOLOv3(nn.Layer):
    """YOLOV3 for detection

    Args:
        ch_in(int): Input channels, default is 3.
        class_num(int): Categories for detection,if dataset is voc, class_num is 20.
        ignore_thresh(float): The ignore threshold to ignore confidence loss.
        valid_thresh(float): Threshold to filter out bounding boxes with low confidence score.
        nms_topk(int): Maximum number of detections to be kept according to the confidences after the filtering
                       detections based on score_threshold.
        nms_posk(int): Number of total bboxes to be kept per image after NMS step. -1 means keeping all bboxes after NMS
                       step.
        nms_thresh (float): The threshold to be used in NMS. Default: 0.3.
        is_train (bool): Set the train mode, default is True.
        load_checkpoint(str): Whether to load checkpoint.
    """
    def __init__(self,
                 ch_in: int = 3,
                 class_num: int = 20,
                 ignore_thresh: float = 0.7,
                 valid_thresh: float = 0.005,
                 nms_topk: int = 400,
                 nms_posk: int = 100,
                 nms_thresh: float = 0.45,
                 is_train: bool = True,
                 load_checkpoint: str = None):
        super(YOLOv3, self).__init__()

        self.is_train = is_train
        self.block = DarkNet53_conv_body(ch_in=ch_in, is_test=not self.is_train)
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        self.class_num = class_num
        self.ignore_thresh = ignore_thresh
        self.valid_thresh = valid_thresh
        self.nms_topk = nms_topk
        self.nms_posk = nms_posk
        self.nms_thresh = nms_thresh
        ch_in_list = [1024, 768, 384]

        for i in range(3):
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(ch_in_list[i], channel=512 // (2**i), is_test=not self.is_train))
            self.yolo_blocks.append(yolo_block)

            num_filters = len(self.anchor_masks[i]) * (self.class_num + 5)
            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                nn.Conv2d(1024 // (2**i),
                          num_filters,
                          1,
                          stride=1,
                          padding=0,
                          weight_attr=paddle.ParamAttr(initializer=Normal(0., 0.02)),
                          bias_attr=paddle.ParamAttr(initializer=Constant(0.0), regularizer=L2Decay(0.))))
            self.block_outputs.append(block_out)

            if i < 2:
                route = self.add_sublayer(
                    "route2_%d" % i,
                    ConvBNLayer(ch_in=512 // (2**i),
                                ch_out=256 // (2**i),
                                filter_size=1,
                                stride=1,
                                padding=0,
                                is_test=(not self.is_train)))
                self.route_blocks_2.append(route)
            self.upsample = Upsample()

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'yolov3_70000.pdparams')
            if not os.path.exists(checkpoint):
                os.system(
                    'wget https://bj.bcebos.com/paddlehub/model/image/object_detection/yolov3_70000.pdparams -O ' \
                    + checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def transform(self, img: paddle.Tensor, size: int):
        if self.is_train:
            transforms = DetectTrainReader()
        else:
            transforms = DetectTestReader()
        return transforms(img, size)

    def get_label_infos(self, file_list: str):
        self.COCO = COCO(file_list)
        label_names = []
        categories = self.COCO.loadCats(self.COCO.getCatIds())
        for category in categories:
            label_names.append(category['name'])
        return label_names

    def forward(self,
                inputs: paddle.Tensor,
                gtbox: paddle.Tensor = None,
                gtlabel: paddle.Tensor = None,
                gtscore: paddle.Tensor = None,
                im_shape: paddle.Tensor = None):

        self.gtbox = gtbox
        self.gtlabel = gtlabel
        self.gtscore = gtscore
        self.im_shape = im_shape
        self.outputs = []
        self.boxes = []
        self.scores = []
        self.losses = []
        self.pred = []
        self.downsample = 32
        blocks = self.block(inputs)
        route = None
        for i, block in enumerate(blocks):
            if i > 0:
                block = paddle.concat([route, block], axis=1)
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            self.outputs.append(block_out)
            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)

        for i, out in enumerate(self.outputs):
            anchor_mask = self.anchor_masks[i]

            if self.is_train:
                loss = F.yolov3_loss(x=out,
                                     gt_box=self.gtbox,
                                     gt_label=self.gtlabel,
                                     gt_score=self.gtscore,
                                     anchors=self.anchors,
                                     anchor_mask=anchor_mask,
                                     class_num=self.class_num,
                                     ignore_thresh=self.ignore_thresh,
                                     downsample_ratio=self.downsample,
                                     use_label_smooth=False)
            else:
                loss = paddle.to_tensor(0.0)
            self.losses.append(paddle.reduce_mean(loss))

            mask_anchors = []
            for m in anchor_mask:
                mask_anchors.append((self.anchors[2 * m]))
                mask_anchors.append(self.anchors[2 * m + 1])

            boxes, scores = F.yolo_box(x=out,
                                       img_size=self.im_shape,
                                       anchors=mask_anchors,
                                       class_num=self.class_num,
                                       conf_thresh=self.valid_thresh,
                                       downsample_ratio=self.downsample,
                                       name="yolo_box" + str(i))

            self.boxes.append(boxes)
            self.scores.append(paddle.transpose(scores, perm=[0, 2, 1]))
            self.downsample //= 2

        for i in range(self.boxes[0].shape[0]):
            yolo_boxes = paddle.unsqueeze(paddle.concat([self.boxes[0][i], self.boxes[1][i], self.boxes[2][i]], axis=0),
                                          0)
            yolo_scores = paddle.unsqueeze(
                paddle.concat([self.scores[0][i], self.scores[1][i], self.scores[2][i]], axis=1), 0)
            pred = F.multiclass_nms(bboxes=yolo_boxes,
                                    scores=yolo_scores,
                                    score_threshold=self.valid_thresh,
                                    nms_top_k=self.nms_topk,
                                    keep_top_k=self.nms_posk,
                                    nms_threshold=self.nms_thresh,
                                    background_label=-1)
            self.pred.append(pred)

        return sum(self.losses), self.pred
