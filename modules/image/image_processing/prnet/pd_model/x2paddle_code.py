import paddle
import math


class TFModel(paddle.nn.Layer):
    def __init__(self):
        super(TFModel, self).__init__()
        self.conv0 = paddle.nn.Conv2D(
            weight_attr='conv0.weight',
            bias_attr=False,
            in_channels=3,
            out_channels=16,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn0 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_Conv_BatchNorm_FusedBatchNorm_resfcn256_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_Conv_BatchNorm_FusedBatchNorm_resfcn256_Conv_BatchNorm_beta',
            moving_mean_name='resfcn256_Conv_BatchNorm_FusedBatchNorm_resfcn256_Conv_BatchNorm_moving_mean',
            moving_variance_name='resfcn256_Conv_BatchNorm_FusedBatchNorm_resfcn256_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(
            weight_attr='conv1.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=32,
            kernel_size=[1, 1],
            stride=2,
            padding='SAME')
        self.conv2 = paddle.nn.Conv2D(
            weight_attr='conv2.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=16,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn1 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu1 = paddle.nn.ReLU()
        self.conv3 = paddle.nn.Conv2D(
            weight_attr='conv3.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=16,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME')
        self.bn2 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu2 = paddle.nn.ReLU()
        self.conv4 = paddle.nn.Conv2D(
            weight_attr='conv4.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=32,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn3 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_BatchNorm_FusedBatchNorm_resfcn256_resBlock_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_BatchNorm_FusedBatchNorm_resfcn256_resBlock_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_BatchNorm_FusedBatchNorm_resfcn256_resBlock_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_BatchNorm_FusedBatchNorm_resfcn256_resBlock_BatchNorm_moving_variance',
            is_test=True)
        self.relu3 = paddle.nn.ReLU()
        self.conv5 = paddle.nn.Conv2D(
            weight_attr='conv5.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=16,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn4 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu4 = paddle.nn.ReLU()
        self.conv6 = paddle.nn.Conv2D(
            weight_attr='conv6.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=16,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn5 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu5 = paddle.nn.ReLU()
        self.conv7 = paddle.nn.Conv2D(
            weight_attr='conv7.weight',
            bias_attr=False,
            in_channels=16,
            out_channels=32,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn6 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu6 = paddle.nn.ReLU()
        self.conv8 = paddle.nn.Conv2D(
            weight_attr='conv8.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=64,
            kernel_size=[1, 1],
            stride=2,
            padding='SAME')
        self.conv9 = paddle.nn.Conv2D(
            weight_attr='conv9.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=32,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn7 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu7 = paddle.nn.ReLU()
        self.conv10 = paddle.nn.Conv2D(
            weight_attr='conv10.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=32,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME')
        self.bn8 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu8 = paddle.nn.ReLU()
        self.conv11 = paddle.nn.Conv2D(
            weight_attr='conv11.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=64,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn9 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_2_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_2_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_2_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_2_BatchNorm_FusedBatchNorm_resfcn256_resBlock_2_BatchNorm_moving_variance',
            is_test=True)
        self.relu9 = paddle.nn.ReLU()
        self.conv12 = paddle.nn.Conv2D(
            weight_attr='conv12.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=32,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn10 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu10 = paddle.nn.ReLU()
        self.conv13 = paddle.nn.Conv2D(
            weight_attr='conv13.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=32,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn11 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu11 = paddle.nn.ReLU()
        self.conv14 = paddle.nn.Conv2D(
            weight_attr='conv14.weight',
            bias_attr=False,
            in_channels=32,
            out_channels=64,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn12 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_3_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_3_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_3_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_3_BatchNorm_FusedBatchNorm_resfcn256_resBlock_3_BatchNorm_moving_variance',
            is_test=True)
        self.relu12 = paddle.nn.ReLU()
        self.conv15 = paddle.nn.Conv2D(
            weight_attr='conv15.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=128,
            kernel_size=[1, 1],
            stride=2,
            padding='SAME')
        self.conv16 = paddle.nn.Conv2D(
            weight_attr='conv16.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=64,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn13 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu13 = paddle.nn.ReLU()
        self.conv17 = paddle.nn.Conv2D(
            weight_attr='conv17.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=64,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME')
        self.bn14 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu14 = paddle.nn.ReLU()
        self.conv18 = paddle.nn.Conv2D(
            weight_attr='conv18.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=128,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn15 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_4_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_4_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_4_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_4_BatchNorm_FusedBatchNorm_resfcn256_resBlock_4_BatchNorm_moving_variance',
            is_test=True)
        self.relu15 = paddle.nn.ReLU()
        self.conv19 = paddle.nn.Conv2D(
            weight_attr='conv19.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=64,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn16 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu16 = paddle.nn.ReLU()
        self.conv20 = paddle.nn.Conv2D(
            weight_attr='conv20.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=64,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn17 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu17 = paddle.nn.ReLU()
        self.conv21 = paddle.nn.Conv2D(
            weight_attr='conv21.weight',
            bias_attr=False,
            in_channels=64,
            out_channels=128,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn18 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_5_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_5_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_5_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_5_BatchNorm_FusedBatchNorm_resfcn256_resBlock_5_BatchNorm_moving_variance',
            is_test=True)
        self.relu18 = paddle.nn.ReLU()
        self.conv22 = paddle.nn.Conv2D(
            weight_attr='conv22.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=256,
            kernel_size=[1, 1],
            stride=2,
            padding='SAME')
        self.conv23 = paddle.nn.Conv2D(
            weight_attr='conv23.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=128,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn19 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu19 = paddle.nn.ReLU()
        self.conv24 = paddle.nn.Conv2D(
            weight_attr='conv24.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=128,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME')
        self.bn20 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu20 = paddle.nn.ReLU()
        self.conv25 = paddle.nn.Conv2D(
            weight_attr='conv25.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=256,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn21 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_6_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_6_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_6_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_6_BatchNorm_FusedBatchNorm_resfcn256_resBlock_6_BatchNorm_moving_variance',
            is_test=True)
        self.relu21 = paddle.nn.ReLU()
        self.conv26 = paddle.nn.Conv2D(
            weight_attr='conv26.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=128,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn22 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu22 = paddle.nn.ReLU()
        self.conv27 = paddle.nn.Conv2D(
            weight_attr='conv27.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=128,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn23 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu23 = paddle.nn.ReLU()
        self.conv28 = paddle.nn.Conv2D(
            weight_attr='conv28.weight',
            bias_attr=False,
            in_channels=128,
            out_channels=256,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn24 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_7_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_7_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_7_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_7_BatchNorm_FusedBatchNorm_resfcn256_resBlock_7_BatchNorm_moving_variance',
            is_test=True)
        self.relu24 = paddle.nn.ReLU()
        self.conv29 = paddle.nn.Conv2D(
            weight_attr='conv29.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=512,
            kernel_size=[1, 1],
            stride=2,
            padding='SAME')
        self.conv30 = paddle.nn.Conv2D(
            weight_attr='conv30.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=256,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn25 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu25 = paddle.nn.ReLU()
        self.conv31 = paddle.nn.Conv2D(
            weight_attr='conv31.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=256,
            kernel_size=[4, 4],
            stride=2,
            padding='SAME')
        self.bn26 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu26 = paddle.nn.ReLU()
        self.conv32 = paddle.nn.Conv2D(
            weight_attr='conv32.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=512,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn27 = paddle.nn.BatchNorm(
            num_channels=512,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_8_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_8_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_8_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_8_BatchNorm_FusedBatchNorm_resfcn256_resBlock_8_BatchNorm_moving_variance',
            is_test=True)
        self.relu27 = paddle.nn.ReLU()
        self.conv33 = paddle.nn.Conv2D(
            weight_attr='conv33.weight',
            bias_attr=False,
            in_channels=512,
            out_channels=256,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn28 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_BatchNorm_moving_variance',
            is_test=True)
        self.relu28 = paddle.nn.ReLU()
        self.conv34 = paddle.nn.Conv2D(
            weight_attr='conv34.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=256,
            kernel_size=[4, 4],
            padding='SAME')
        self.bn29 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_1_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_Conv_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu29 = paddle.nn.ReLU()
        self.conv35 = paddle.nn.Conv2D(
            weight_attr='conv35.weight',
            bias_attr=False,
            in_channels=256,
            out_channels=512,
            kernel_size=[1, 1],
            padding='SAME')
        self.bn30 = paddle.nn.BatchNorm(
            num_channels=512,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_resBlock_9_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_BatchNorm_gamma',
            bias_attr='resfcn256_resBlock_9_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_BatchNorm_beta',
            moving_mean_name='resfcn256_resBlock_9_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_resBlock_9_BatchNorm_FusedBatchNorm_resfcn256_resBlock_9_BatchNorm_moving_variance',
            is_test=True)
        self.relu30 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_conv2d_transpose_conv36_weight = self.create_parameter(
            shape=(512, 512, 4, 4), attr='conv36.weight')
        self.bn31 = paddle.nn.BatchNorm(
            num_channels=512,
            epsilon=0.0010000000474974513,
            param_attr='resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_BatchNorm_gamma',
            bias_attr='resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_BatchNorm_moving_variance',
            is_test=True)
        self.relu31 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_1_conv2d_transpose_conv37_weight = self.create_parameter(
            shape=(512, 256, 4, 4), attr='conv37.weight')
        self.bn32 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_1_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_1_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_1_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_1_BatchNorm_moving_variance',
            is_test=True)
        self.relu32 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_2_conv2d_transpose_conv38_weight = self.create_parameter(
            shape=(256, 256, 4, 4), attr='conv38.weight')
        self.bn33 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_2_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_2_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_2_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_2_BatchNorm_moving_variance',
            is_test=True)
        self.relu33 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_3_conv2d_transpose_conv39_weight = self.create_parameter(
            shape=(256, 256, 4, 4), attr='conv39.weight')
        self.bn34 = paddle.nn.BatchNorm(
            num_channels=256,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_3_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_3_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_3_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_3_BatchNorm_moving_variance',
            is_test=True)
        self.relu34 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_4_conv2d_transpose_conv40_weight = self.create_parameter(
            shape=(256, 128, 4, 4), attr='conv40.weight')
        self.bn35 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_4_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_4_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_4_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_4_BatchNorm_moving_variance',
            is_test=True)
        self.relu35 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_5_conv2d_transpose_conv41_weight = self.create_parameter(
            shape=(128, 128, 4, 4), attr='conv41.weight')
        self.bn36 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_5_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_5_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_5_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_5_BatchNorm_moving_variance',
            is_test=True)
        self.relu36 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_6_conv2d_transpose_conv42_weight = self.create_parameter(
            shape=(128, 128, 4, 4), attr='conv42.weight')
        self.bn37 = paddle.nn.BatchNorm(
            num_channels=128,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_6_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_6_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_6_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_6_BatchNorm_moving_variance',
            is_test=True)
        self.relu37 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_7_conv2d_transpose_conv43_weight = self.create_parameter(
            shape=(128, 64, 4, 4), attr='conv43.weight')
        self.bn38 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_7_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_7_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_7_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_7_BatchNorm_moving_variance',
            is_test=True)
        self.relu38 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_8_conv2d_transpose_conv44_weight = self.create_parameter(
            shape=(64, 64, 4, 4), attr='conv44.weight')
        self.bn39 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_8_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_8_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_8_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_8_BatchNorm_moving_variance',
            is_test=True)
        self.relu39 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_9_conv2d_transpose_conv45_weight = self.create_parameter(
            shape=(64, 64, 4, 4), attr='conv45.weight')
        self.bn40 = paddle.nn.BatchNorm(
            num_channels=64,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_9_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_9_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_9_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_9_BatchNorm_moving_variance',
            is_test=True)
        self.relu40 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_10_conv2d_transpose_conv46_weight = self.create_parameter(
            shape=(64, 32, 4, 4), attr='conv46.weight')
        self.bn41 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_10_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_10_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_10_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_10_BatchNorm_moving_variance',
            is_test=True)
        self.relu41 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_11_conv2d_transpose_conv47_weight = self.create_parameter(
            shape=(32, 32, 4, 4), attr='conv47.weight')
        self.bn42 = paddle.nn.BatchNorm(
            num_channels=32,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_11_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_11_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_11_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_11_BatchNorm_moving_variance',
            is_test=True)
        self.relu42 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_12_conv2d_transpose_conv48_weight = self.create_parameter(
            shape=(32, 16, 4, 4), attr='conv48.weight')
        self.bn43 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_12_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_12_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_12_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_12_BatchNorm_moving_variance',
            is_test=True)
        self.relu43 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_13_conv2d_transpose_conv49_weight = self.create_parameter(
            shape=(16, 16, 4, 4), attr='conv49.weight')
        self.bn44 = paddle.nn.BatchNorm(
            num_channels=16,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_13_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_13_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_13_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_13_BatchNorm_moving_variance',
            is_test=True)
        self.relu44 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_14_conv2d_transpose_conv50_weight = self.create_parameter(
            shape=(16, 3, 4, 4), attr='conv50.weight')
        self.bn45 = paddle.nn.BatchNorm(
            num_channels=3,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_14_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_14_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_14_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_14_BatchNorm_moving_variance',
            is_test=True)
        self.relu45 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_15_conv2d_transpose_conv51_weight = self.create_parameter(
            shape=(3, 3, 4, 4), attr='conv51.weight')
        self.bn46 = paddle.nn.BatchNorm(
            num_channels=3,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_15_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_15_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_15_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_15_BatchNorm_moving_variance',
            is_test=True)
        self.relu46 = paddle.nn.ReLU()
        self.resfcn256_Conv2d_transpose_16_conv2d_transpose_conv52_weight = self.create_parameter(
            shape=(3, 3, 4, 4), attr='conv52.weight')
        self.bn47 = paddle.nn.BatchNorm(
            num_channels=3,
            epsilon=0.0010000000474974513,
            param_attr=
            'resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_16_BatchNorm_gamma',
            bias_attr=
            'resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_16_BatchNorm_beta',
            moving_mean_name=
            'resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_16_BatchNorm_moving_mean',
            moving_variance_name=
            'resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm_resfcn256_Conv2d_transpose_16_BatchNorm_moving_variance',
            is_test=True)
        self.sigmoid0 = paddle.nn.Sigmoid()

    def forward(self, Placeholder):
        resfcn256_Conv2d_transpose_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=512)
        resfcn256_Conv2d_transpose_1_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_1_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_1_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=256)
        resfcn256_Conv2d_transpose_2_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_2_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_2_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=256)
        resfcn256_Conv2d_transpose_3_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_3_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_3_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=256)
        resfcn256_Conv2d_transpose_4_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_4_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_4_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=128)
        resfcn256_Conv2d_transpose_5_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_5_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_5_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=128)
        resfcn256_Conv2d_transpose_6_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_6_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_6_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=128)
        resfcn256_Conv2d_transpose_7_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_7_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_7_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=64)
        resfcn256_Conv2d_transpose_8_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_8_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_8_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=64)
        resfcn256_Conv2d_transpose_9_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_9_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_9_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=64)
        resfcn256_Conv2d_transpose_10_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_10_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_10_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=32)
        resfcn256_Conv2d_transpose_11_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_11_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_11_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=32)
        resfcn256_Conv2d_transpose_12_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_12_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=2)
        resfcn256_Conv2d_transpose_12_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=16)
        resfcn256_Conv2d_transpose_13_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_13_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_13_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=16)
        resfcn256_Conv2d_transpose_14_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_14_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_14_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=3)
        resfcn256_Conv2d_transpose_15_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_15_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_15_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=3)
        resfcn256_Conv2d_transpose_16_mul_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_16_mul_1_y = paddle.full(dtype='int32', shape=[1], fill_value=1)
        resfcn256_Conv2d_transpose_16_stack_3 = paddle.full(dtype='int32', shape=[1], fill_value=3)
        conv2d_transpose_0 = paddle.transpose(x=Placeholder, perm=[0, 3, 1, 2])
        resfcn256_Conv_Conv2D = self.conv0(conv2d_transpose_0)
        resfcn256_Conv_BatchNorm_FusedBatchNorm = self.bn0(resfcn256_Conv_Conv2D)
        resfcn256_Conv_Relu = self.relu0(resfcn256_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_shortcut_Conv2D = self.conv1(resfcn256_Conv_Relu)
        resfcn256_resBlock_Conv_Conv2D = self.conv2(resfcn256_Conv_Relu)
        resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm = self.bn1(resfcn256_resBlock_Conv_Conv2D)
        resfcn256_resBlock_Conv_Relu = self.relu1(resfcn256_resBlock_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_Conv_1_Conv2D = self.conv3(resfcn256_resBlock_Conv_Relu)
        resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm = self.bn2(resfcn256_resBlock_Conv_1_Conv2D)
        resfcn256_resBlock_Conv_1_Relu = self.relu2(resfcn256_resBlock_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_Conv_2_Conv2D = self.conv4(resfcn256_resBlock_Conv_1_Relu)
        resfcn256_resBlock_add = paddle.add(x=resfcn256_resBlock_Conv_2_Conv2D, y=resfcn256_resBlock_shortcut_Conv2D)
        resfcn256_resBlock_BatchNorm_FusedBatchNorm = self.bn3(resfcn256_resBlock_add)
        resfcn256_resBlock_Relu = self.relu3(resfcn256_resBlock_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_1_Conv_Conv2D = self.conv5(resfcn256_resBlock_Relu)
        resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm = self.bn4(resfcn256_resBlock_1_Conv_Conv2D)
        resfcn256_resBlock_1_Conv_Relu = self.relu4(resfcn256_resBlock_1_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_1_Conv_1_Conv2D = self.conv6(resfcn256_resBlock_1_Conv_Relu)
        resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm = self.bn5(resfcn256_resBlock_1_Conv_1_Conv2D)
        resfcn256_resBlock_1_Conv_1_Relu = self.relu5(resfcn256_resBlock_1_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_1_Conv_2_Conv2D = self.conv7(resfcn256_resBlock_1_Conv_1_Relu)
        resfcn256_resBlock_1_add = paddle.add(x=resfcn256_resBlock_1_Conv_2_Conv2D, y=resfcn256_resBlock_Relu)
        resfcn256_resBlock_1_BatchNorm_FusedBatchNorm = self.bn6(resfcn256_resBlock_1_add)
        resfcn256_resBlock_1_Relu = self.relu6(resfcn256_resBlock_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_2_shortcut_Conv2D = self.conv8(resfcn256_resBlock_1_Relu)
        resfcn256_resBlock_2_Conv_Conv2D = self.conv9(resfcn256_resBlock_1_Relu)
        resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm = self.bn7(resfcn256_resBlock_2_Conv_Conv2D)
        resfcn256_resBlock_2_Conv_Relu = self.relu7(resfcn256_resBlock_2_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_2_Conv_1_Conv2D = self.conv10(resfcn256_resBlock_2_Conv_Relu)
        resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm = self.bn8(resfcn256_resBlock_2_Conv_1_Conv2D)
        resfcn256_resBlock_2_Conv_1_Relu = self.relu8(resfcn256_resBlock_2_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_2_Conv_2_Conv2D = self.conv11(resfcn256_resBlock_2_Conv_1_Relu)
        resfcn256_resBlock_2_add = paddle.add(
            x=resfcn256_resBlock_2_Conv_2_Conv2D, y=resfcn256_resBlock_2_shortcut_Conv2D)
        resfcn256_resBlock_2_BatchNorm_FusedBatchNorm = self.bn9(resfcn256_resBlock_2_add)
        resfcn256_resBlock_2_Relu = self.relu9(resfcn256_resBlock_2_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_3_Conv_Conv2D = self.conv12(resfcn256_resBlock_2_Relu)
        resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm = self.bn10(resfcn256_resBlock_3_Conv_Conv2D)
        resfcn256_resBlock_3_Conv_Relu = self.relu10(resfcn256_resBlock_3_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_3_Conv_1_Conv2D = self.conv13(resfcn256_resBlock_3_Conv_Relu)
        resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm = self.bn11(resfcn256_resBlock_3_Conv_1_Conv2D)
        resfcn256_resBlock_3_Conv_1_Relu = self.relu11(resfcn256_resBlock_3_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_3_Conv_2_Conv2D = self.conv14(resfcn256_resBlock_3_Conv_1_Relu)
        resfcn256_resBlock_3_add = paddle.add(x=resfcn256_resBlock_3_Conv_2_Conv2D, y=resfcn256_resBlock_2_Relu)
        resfcn256_resBlock_3_BatchNorm_FusedBatchNorm = self.bn12(resfcn256_resBlock_3_add)
        resfcn256_resBlock_3_Relu = self.relu12(resfcn256_resBlock_3_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_4_shortcut_Conv2D = self.conv15(resfcn256_resBlock_3_Relu)
        resfcn256_resBlock_4_Conv_Conv2D = self.conv16(resfcn256_resBlock_3_Relu)
        resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm = self.bn13(resfcn256_resBlock_4_Conv_Conv2D)
        resfcn256_resBlock_4_Conv_Relu = self.relu13(resfcn256_resBlock_4_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_4_Conv_1_Conv2D = self.conv17(resfcn256_resBlock_4_Conv_Relu)
        resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm = self.bn14(resfcn256_resBlock_4_Conv_1_Conv2D)
        resfcn256_resBlock_4_Conv_1_Relu = self.relu14(resfcn256_resBlock_4_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_4_Conv_2_Conv2D = self.conv18(resfcn256_resBlock_4_Conv_1_Relu)
        resfcn256_resBlock_4_add = paddle.add(
            x=resfcn256_resBlock_4_Conv_2_Conv2D, y=resfcn256_resBlock_4_shortcut_Conv2D)
        resfcn256_resBlock_4_BatchNorm_FusedBatchNorm = self.bn15(resfcn256_resBlock_4_add)
        resfcn256_resBlock_4_Relu = self.relu15(resfcn256_resBlock_4_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_5_Conv_Conv2D = self.conv19(resfcn256_resBlock_4_Relu)
        resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm = self.bn16(resfcn256_resBlock_5_Conv_Conv2D)
        resfcn256_resBlock_5_Conv_Relu = self.relu16(resfcn256_resBlock_5_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_5_Conv_1_Conv2D = self.conv20(resfcn256_resBlock_5_Conv_Relu)
        resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm = self.bn17(resfcn256_resBlock_5_Conv_1_Conv2D)
        resfcn256_resBlock_5_Conv_1_Relu = self.relu17(resfcn256_resBlock_5_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_5_Conv_2_Conv2D = self.conv21(resfcn256_resBlock_5_Conv_1_Relu)
        resfcn256_resBlock_5_add = paddle.add(x=resfcn256_resBlock_5_Conv_2_Conv2D, y=resfcn256_resBlock_4_Relu)
        resfcn256_resBlock_5_BatchNorm_FusedBatchNorm = self.bn18(resfcn256_resBlock_5_add)
        resfcn256_resBlock_5_Relu = self.relu18(resfcn256_resBlock_5_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_6_shortcut_Conv2D = self.conv22(resfcn256_resBlock_5_Relu)
        resfcn256_resBlock_6_Conv_Conv2D = self.conv23(resfcn256_resBlock_5_Relu)
        resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm = self.bn19(resfcn256_resBlock_6_Conv_Conv2D)
        resfcn256_resBlock_6_Conv_Relu = self.relu19(resfcn256_resBlock_6_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_6_Conv_1_Conv2D = self.conv24(resfcn256_resBlock_6_Conv_Relu)
        resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm = self.bn20(resfcn256_resBlock_6_Conv_1_Conv2D)
        resfcn256_resBlock_6_Conv_1_Relu = self.relu20(resfcn256_resBlock_6_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_6_Conv_2_Conv2D = self.conv25(resfcn256_resBlock_6_Conv_1_Relu)
        resfcn256_resBlock_6_add = paddle.add(
            x=resfcn256_resBlock_6_Conv_2_Conv2D, y=resfcn256_resBlock_6_shortcut_Conv2D)
        resfcn256_resBlock_6_BatchNorm_FusedBatchNorm = self.bn21(resfcn256_resBlock_6_add)
        resfcn256_resBlock_6_Relu = self.relu21(resfcn256_resBlock_6_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_7_Conv_Conv2D = self.conv26(resfcn256_resBlock_6_Relu)
        resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm = self.bn22(resfcn256_resBlock_7_Conv_Conv2D)
        resfcn256_resBlock_7_Conv_Relu = self.relu22(resfcn256_resBlock_7_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_7_Conv_1_Conv2D = self.conv27(resfcn256_resBlock_7_Conv_Relu)
        resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm = self.bn23(resfcn256_resBlock_7_Conv_1_Conv2D)
        resfcn256_resBlock_7_Conv_1_Relu = self.relu23(resfcn256_resBlock_7_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_7_Conv_2_Conv2D = self.conv28(resfcn256_resBlock_7_Conv_1_Relu)
        resfcn256_resBlock_7_add = paddle.add(x=resfcn256_resBlock_7_Conv_2_Conv2D, y=resfcn256_resBlock_6_Relu)
        resfcn256_resBlock_7_BatchNorm_FusedBatchNorm = self.bn24(resfcn256_resBlock_7_add)
        resfcn256_resBlock_7_Relu = self.relu24(resfcn256_resBlock_7_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_8_shortcut_Conv2D = self.conv29(resfcn256_resBlock_7_Relu)
        resfcn256_resBlock_8_Conv_Conv2D = self.conv30(resfcn256_resBlock_7_Relu)
        resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm = self.bn25(resfcn256_resBlock_8_Conv_Conv2D)
        resfcn256_resBlock_8_Conv_Relu = self.relu25(resfcn256_resBlock_8_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_8_Conv_1_Conv2D = self.conv31(resfcn256_resBlock_8_Conv_Relu)
        resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm = self.bn26(resfcn256_resBlock_8_Conv_1_Conv2D)
        resfcn256_resBlock_8_Conv_1_Relu = self.relu26(resfcn256_resBlock_8_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_8_Conv_2_Conv2D = self.conv32(resfcn256_resBlock_8_Conv_1_Relu)
        resfcn256_resBlock_8_add = paddle.add(
            x=resfcn256_resBlock_8_Conv_2_Conv2D, y=resfcn256_resBlock_8_shortcut_Conv2D)
        resfcn256_resBlock_8_BatchNorm_FusedBatchNorm = self.bn27(resfcn256_resBlock_8_add)
        resfcn256_resBlock_8_Relu = self.relu27(resfcn256_resBlock_8_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_9_Conv_Conv2D = self.conv33(resfcn256_resBlock_8_Relu)
        resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm = self.bn28(resfcn256_resBlock_9_Conv_Conv2D)
        resfcn256_resBlock_9_Conv_Relu = self.relu28(resfcn256_resBlock_9_Conv_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_9_Conv_1_Conv2D = self.conv34(resfcn256_resBlock_9_Conv_Relu)
        resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm = self.bn29(resfcn256_resBlock_9_Conv_1_Conv2D)
        resfcn256_resBlock_9_Conv_1_Relu = self.relu29(resfcn256_resBlock_9_Conv_1_BatchNorm_FusedBatchNorm)
        resfcn256_resBlock_9_Conv_2_Conv2D = self.conv35(resfcn256_resBlock_9_Conv_1_Relu)
        resfcn256_resBlock_9_add = paddle.add(x=resfcn256_resBlock_9_Conv_2_Conv2D, y=resfcn256_resBlock_8_Relu)
        resfcn256_resBlock_9_BatchNorm_FusedBatchNorm = self.bn30(resfcn256_resBlock_9_add)
        resfcn256_resBlock_9_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_resBlock_9_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_resBlock_9_Relu = self.relu30(resfcn256_resBlock_9_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_Shape = paddle.shape(input=resfcn256_resBlock_9_Relu)
        resfcn256_Conv2d_transpose_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_strided_slice_1, y=resfcn256_Conv2d_transpose_mul_y)
        resfcn256_Conv2d_transpose_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_strided_slice_2, y=resfcn256_Conv2d_transpose_mul_1_y)
        resfcn256_Conv2d_transpose_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_strided_slice, resfcn256_Conv2d_transpose_mul, resfcn256_Conv2d_transpose_mul_1,
            resfcn256_Conv2d_transpose_stack_3
        ])
        resfcn256_Conv2d_transpose_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_stack, shape=[-1])
        conv2dbackpropinput_transpose_0 = paddle.transpose(x=resfcn256_resBlock_9_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_conv2d_transpose_conv36_weight = self.resfcn256_Conv2d_transpose_conv2d_transpose_conv36_weight
        resfcn256_Conv2d_transpose_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_0,
            weight=resfcn256_Conv2d_transpose_conv2d_transpose_conv36_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[8, 8])
        resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm = self.bn31(resfcn256_Conv2d_transpose_conv2d_transpose)
        resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_Relu = self.relu31(resfcn256_Conv2d_transpose_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_1_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_Relu)
        resfcn256_Conv2d_transpose_1_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_1_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_1_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_1_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_1_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_1_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_1_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_1_strided_slice_1, y=resfcn256_Conv2d_transpose_1_mul_y)
        resfcn256_Conv2d_transpose_1_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_1_strided_slice_2, y=resfcn256_Conv2d_transpose_1_mul_1_y)
        resfcn256_Conv2d_transpose_1_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_1_strided_slice, resfcn256_Conv2d_transpose_1_mul,
            resfcn256_Conv2d_transpose_1_mul_1, resfcn256_Conv2d_transpose_1_stack_3
        ])
        resfcn256_Conv2d_transpose_1_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_1_stack, shape=[-1])
        conv2dbackpropinput_transpose_1 = paddle.transpose(x=resfcn256_Conv2d_transpose_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_1_conv2d_transpose_conv37_weight = self.resfcn256_Conv2d_transpose_1_conv2d_transpose_conv37_weight
        resfcn256_Conv2d_transpose_1_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_1,
            weight=resfcn256_Conv2d_transpose_1_conv2d_transpose_conv37_weight,
            stride=[2, 2],
            dilation=[1, 1],
            padding='SAME',
            output_size=[16, 16])
        resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm = self.bn32(resfcn256_Conv2d_transpose_1_conv2d_transpose)
        resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_1_Relu = self.relu32(resfcn256_Conv2d_transpose_1_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_2_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_1_Relu)
        resfcn256_Conv2d_transpose_2_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_2_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_2_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_2_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_2_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_2_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_2_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_2_strided_slice_1, y=resfcn256_Conv2d_transpose_2_mul_y)
        resfcn256_Conv2d_transpose_2_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_2_strided_slice_2, y=resfcn256_Conv2d_transpose_2_mul_1_y)
        resfcn256_Conv2d_transpose_2_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_2_strided_slice, resfcn256_Conv2d_transpose_2_mul,
            resfcn256_Conv2d_transpose_2_mul_1, resfcn256_Conv2d_transpose_2_stack_3
        ])
        resfcn256_Conv2d_transpose_2_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_2_stack, shape=[-1])
        conv2dbackpropinput_transpose_2 = paddle.transpose(x=resfcn256_Conv2d_transpose_1_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_2_conv2d_transpose_conv38_weight = self.resfcn256_Conv2d_transpose_2_conv2d_transpose_conv38_weight
        resfcn256_Conv2d_transpose_2_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_2,
            weight=resfcn256_Conv2d_transpose_2_conv2d_transpose_conv38_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[16, 16])
        resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm = self.bn33(resfcn256_Conv2d_transpose_2_conv2d_transpose)
        resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_2_Relu = self.relu33(resfcn256_Conv2d_transpose_2_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_3_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_2_Relu)
        resfcn256_Conv2d_transpose_3_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_3_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_3_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_3_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_3_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_3_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_3_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_3_strided_slice_1, y=resfcn256_Conv2d_transpose_3_mul_y)
        resfcn256_Conv2d_transpose_3_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_3_strided_slice_2, y=resfcn256_Conv2d_transpose_3_mul_1_y)
        resfcn256_Conv2d_transpose_3_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_3_strided_slice, resfcn256_Conv2d_transpose_3_mul,
            resfcn256_Conv2d_transpose_3_mul_1, resfcn256_Conv2d_transpose_3_stack_3
        ])
        resfcn256_Conv2d_transpose_3_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_3_stack, shape=[-1])
        conv2dbackpropinput_transpose_3 = paddle.transpose(x=resfcn256_Conv2d_transpose_2_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_3_conv2d_transpose_conv39_weight = self.resfcn256_Conv2d_transpose_3_conv2d_transpose_conv39_weight
        resfcn256_Conv2d_transpose_3_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_3,
            weight=resfcn256_Conv2d_transpose_3_conv2d_transpose_conv39_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[16, 16])
        resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm = self.bn34(resfcn256_Conv2d_transpose_3_conv2d_transpose)
        resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_3_Relu = self.relu34(resfcn256_Conv2d_transpose_3_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_4_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_3_Relu)
        resfcn256_Conv2d_transpose_4_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_4_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_4_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_4_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_4_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_4_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_4_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_4_strided_slice_1, y=resfcn256_Conv2d_transpose_4_mul_y)
        resfcn256_Conv2d_transpose_4_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_4_strided_slice_2, y=resfcn256_Conv2d_transpose_4_mul_1_y)
        resfcn256_Conv2d_transpose_4_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_4_strided_slice, resfcn256_Conv2d_transpose_4_mul,
            resfcn256_Conv2d_transpose_4_mul_1, resfcn256_Conv2d_transpose_4_stack_3
        ])
        resfcn256_Conv2d_transpose_4_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_4_stack, shape=[-1])
        conv2dbackpropinput_transpose_4 = paddle.transpose(x=resfcn256_Conv2d_transpose_3_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_4_conv2d_transpose_conv40_weight = self.resfcn256_Conv2d_transpose_4_conv2d_transpose_conv40_weight
        resfcn256_Conv2d_transpose_4_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_4,
            weight=resfcn256_Conv2d_transpose_4_conv2d_transpose_conv40_weight,
            stride=[2, 2],
            dilation=[1, 1],
            padding='SAME',
            output_size=[32, 32])
        resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm = self.bn35(resfcn256_Conv2d_transpose_4_conv2d_transpose)
        resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_4_Relu = self.relu35(resfcn256_Conv2d_transpose_4_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_5_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_4_Relu)
        resfcn256_Conv2d_transpose_5_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_5_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_5_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_5_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_5_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_5_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_5_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_5_strided_slice_1, y=resfcn256_Conv2d_transpose_5_mul_y)
        resfcn256_Conv2d_transpose_5_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_5_strided_slice_2, y=resfcn256_Conv2d_transpose_5_mul_1_y)
        resfcn256_Conv2d_transpose_5_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_5_strided_slice, resfcn256_Conv2d_transpose_5_mul,
            resfcn256_Conv2d_transpose_5_mul_1, resfcn256_Conv2d_transpose_5_stack_3
        ])
        resfcn256_Conv2d_transpose_5_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_5_stack, shape=[-1])
        conv2dbackpropinput_transpose_5 = paddle.transpose(x=resfcn256_Conv2d_transpose_4_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_5_conv2d_transpose_conv41_weight = self.resfcn256_Conv2d_transpose_5_conv2d_transpose_conv41_weight
        resfcn256_Conv2d_transpose_5_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_5,
            weight=resfcn256_Conv2d_transpose_5_conv2d_transpose_conv41_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[32, 32])
        resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm = self.bn36(resfcn256_Conv2d_transpose_5_conv2d_transpose)
        resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_5_Relu = self.relu36(resfcn256_Conv2d_transpose_5_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_6_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_5_Relu)
        resfcn256_Conv2d_transpose_6_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_6_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_6_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_6_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_6_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_6_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_6_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_6_strided_slice_1, y=resfcn256_Conv2d_transpose_6_mul_y)
        resfcn256_Conv2d_transpose_6_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_6_strided_slice_2, y=resfcn256_Conv2d_transpose_6_mul_1_y)
        resfcn256_Conv2d_transpose_6_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_6_strided_slice, resfcn256_Conv2d_transpose_6_mul,
            resfcn256_Conv2d_transpose_6_mul_1, resfcn256_Conv2d_transpose_6_stack_3
        ])
        resfcn256_Conv2d_transpose_6_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_6_stack, shape=[-1])
        conv2dbackpropinput_transpose_6 = paddle.transpose(x=resfcn256_Conv2d_transpose_5_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_6_conv2d_transpose_conv42_weight = self.resfcn256_Conv2d_transpose_6_conv2d_transpose_conv42_weight
        resfcn256_Conv2d_transpose_6_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_6,
            weight=resfcn256_Conv2d_transpose_6_conv2d_transpose_conv42_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[32, 32])
        resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm = self.bn37(resfcn256_Conv2d_transpose_6_conv2d_transpose)
        resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_6_Relu = self.relu37(resfcn256_Conv2d_transpose_6_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_7_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_6_Relu)
        resfcn256_Conv2d_transpose_7_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_7_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_7_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_7_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_7_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_7_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_7_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_7_strided_slice_1, y=resfcn256_Conv2d_transpose_7_mul_y)
        resfcn256_Conv2d_transpose_7_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_7_strided_slice_2, y=resfcn256_Conv2d_transpose_7_mul_1_y)
        resfcn256_Conv2d_transpose_7_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_7_strided_slice, resfcn256_Conv2d_transpose_7_mul,
            resfcn256_Conv2d_transpose_7_mul_1, resfcn256_Conv2d_transpose_7_stack_3
        ])
        resfcn256_Conv2d_transpose_7_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_7_stack, shape=[-1])
        conv2dbackpropinput_transpose_7 = paddle.transpose(x=resfcn256_Conv2d_transpose_6_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_7_conv2d_transpose_conv43_weight = self.resfcn256_Conv2d_transpose_7_conv2d_transpose_conv43_weight
        resfcn256_Conv2d_transpose_7_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_7,
            weight=resfcn256_Conv2d_transpose_7_conv2d_transpose_conv43_weight,
            stride=[2, 2],
            dilation=[1, 1],
            padding='SAME',
            output_size=[64, 64])
        resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm = self.bn38(resfcn256_Conv2d_transpose_7_conv2d_transpose)
        resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_7_Relu = self.relu38(resfcn256_Conv2d_transpose_7_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_8_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_7_Relu)
        resfcn256_Conv2d_transpose_8_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_8_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_8_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_8_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_8_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_8_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_8_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_8_strided_slice_1, y=resfcn256_Conv2d_transpose_8_mul_y)
        resfcn256_Conv2d_transpose_8_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_8_strided_slice_2, y=resfcn256_Conv2d_transpose_8_mul_1_y)
        resfcn256_Conv2d_transpose_8_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_8_strided_slice, resfcn256_Conv2d_transpose_8_mul,
            resfcn256_Conv2d_transpose_8_mul_1, resfcn256_Conv2d_transpose_8_stack_3
        ])
        resfcn256_Conv2d_transpose_8_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_8_stack, shape=[-1])
        conv2dbackpropinput_transpose_8 = paddle.transpose(x=resfcn256_Conv2d_transpose_7_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_8_conv2d_transpose_conv44_weight = self.resfcn256_Conv2d_transpose_8_conv2d_transpose_conv44_weight
        resfcn256_Conv2d_transpose_8_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_8,
            weight=resfcn256_Conv2d_transpose_8_conv2d_transpose_conv44_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[64, 64])
        resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm = self.bn39(resfcn256_Conv2d_transpose_8_conv2d_transpose)
        resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_8_Relu = self.relu39(resfcn256_Conv2d_transpose_8_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_9_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_8_Relu)
        resfcn256_Conv2d_transpose_9_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_9_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_9_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_9_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_9_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_9_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_9_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_9_strided_slice_1, y=resfcn256_Conv2d_transpose_9_mul_y)
        resfcn256_Conv2d_transpose_9_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_9_strided_slice_2, y=resfcn256_Conv2d_transpose_9_mul_1_y)
        resfcn256_Conv2d_transpose_9_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_9_strided_slice, resfcn256_Conv2d_transpose_9_mul,
            resfcn256_Conv2d_transpose_9_mul_1, resfcn256_Conv2d_transpose_9_stack_3
        ])
        resfcn256_Conv2d_transpose_9_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_9_stack, shape=[-1])
        conv2dbackpropinput_transpose_9 = paddle.transpose(x=resfcn256_Conv2d_transpose_8_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_9_conv2d_transpose_conv45_weight = self.resfcn256_Conv2d_transpose_9_conv2d_transpose_conv45_weight
        resfcn256_Conv2d_transpose_9_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_9,
            weight=resfcn256_Conv2d_transpose_9_conv2d_transpose_conv45_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[64, 64])
        resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm = self.bn40(resfcn256_Conv2d_transpose_9_conv2d_transpose)
        resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_9_Relu = self.relu40(resfcn256_Conv2d_transpose_9_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_10_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_9_Relu)
        resfcn256_Conv2d_transpose_10_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_10_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_10_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_10_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_10_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_10_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_10_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_10_strided_slice_1, y=resfcn256_Conv2d_transpose_10_mul_y)
        resfcn256_Conv2d_transpose_10_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_10_strided_slice_2, y=resfcn256_Conv2d_transpose_10_mul_1_y)
        resfcn256_Conv2d_transpose_10_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_10_strided_slice, resfcn256_Conv2d_transpose_10_mul,
            resfcn256_Conv2d_transpose_10_mul_1, resfcn256_Conv2d_transpose_10_stack_3
        ])
        resfcn256_Conv2d_transpose_10_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_10_stack, shape=[-1])
        conv2dbackpropinput_transpose_10 = paddle.transpose(x=resfcn256_Conv2d_transpose_9_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_10_conv2d_transpose_conv46_weight = self.resfcn256_Conv2d_transpose_10_conv2d_transpose_conv46_weight
        resfcn256_Conv2d_transpose_10_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_10,
            weight=resfcn256_Conv2d_transpose_10_conv2d_transpose_conv46_weight,
            stride=[2, 2],
            dilation=[1, 1],
            padding='SAME',
            output_size=[128, 128])
        resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm = self.bn41(
            resfcn256_Conv2d_transpose_10_conv2d_transpose)
        resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_10_Relu = self.relu41(resfcn256_Conv2d_transpose_10_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_11_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_10_Relu)
        resfcn256_Conv2d_transpose_11_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_11_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_11_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_11_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_11_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_11_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_11_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_11_strided_slice_1, y=resfcn256_Conv2d_transpose_11_mul_y)
        resfcn256_Conv2d_transpose_11_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_11_strided_slice_2, y=resfcn256_Conv2d_transpose_11_mul_1_y)
        resfcn256_Conv2d_transpose_11_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_11_strided_slice, resfcn256_Conv2d_transpose_11_mul,
            resfcn256_Conv2d_transpose_11_mul_1, resfcn256_Conv2d_transpose_11_stack_3
        ])
        resfcn256_Conv2d_transpose_11_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_11_stack, shape=[-1])
        conv2dbackpropinput_transpose_11 = paddle.transpose(x=resfcn256_Conv2d_transpose_10_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_11_conv2d_transpose_conv47_weight = self.resfcn256_Conv2d_transpose_11_conv2d_transpose_conv47_weight
        resfcn256_Conv2d_transpose_11_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_11,
            weight=resfcn256_Conv2d_transpose_11_conv2d_transpose_conv47_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[128, 128])
        resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm = self.bn42(
            resfcn256_Conv2d_transpose_11_conv2d_transpose)
        resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_11_Relu = self.relu42(resfcn256_Conv2d_transpose_11_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_12_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_11_Relu)
        resfcn256_Conv2d_transpose_12_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_12_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_12_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_12_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_12_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_12_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_12_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_12_strided_slice_1, y=resfcn256_Conv2d_transpose_12_mul_y)
        resfcn256_Conv2d_transpose_12_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_12_strided_slice_2, y=resfcn256_Conv2d_transpose_12_mul_1_y)
        resfcn256_Conv2d_transpose_12_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_12_strided_slice, resfcn256_Conv2d_transpose_12_mul,
            resfcn256_Conv2d_transpose_12_mul_1, resfcn256_Conv2d_transpose_12_stack_3
        ])
        resfcn256_Conv2d_transpose_12_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_12_stack, shape=[-1])
        conv2dbackpropinput_transpose_12 = paddle.transpose(x=resfcn256_Conv2d_transpose_11_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_12_conv2d_transpose_conv48_weight = self.resfcn256_Conv2d_transpose_12_conv2d_transpose_conv48_weight
        resfcn256_Conv2d_transpose_12_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_12,
            weight=resfcn256_Conv2d_transpose_12_conv2d_transpose_conv48_weight,
            stride=[2, 2],
            dilation=[1, 1],
            padding='SAME',
            output_size=[256, 256])
        resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm = self.bn43(
            resfcn256_Conv2d_transpose_12_conv2d_transpose)
        resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_12_Relu = self.relu43(resfcn256_Conv2d_transpose_12_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_13_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_12_Relu)
        resfcn256_Conv2d_transpose_13_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_13_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_13_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_13_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_13_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_13_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_13_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_13_strided_slice_1, y=resfcn256_Conv2d_transpose_13_mul_y)
        resfcn256_Conv2d_transpose_13_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_13_strided_slice_2, y=resfcn256_Conv2d_transpose_13_mul_1_y)
        resfcn256_Conv2d_transpose_13_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_13_strided_slice, resfcn256_Conv2d_transpose_13_mul,
            resfcn256_Conv2d_transpose_13_mul_1, resfcn256_Conv2d_transpose_13_stack_3
        ])
        resfcn256_Conv2d_transpose_13_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_13_stack, shape=[-1])
        conv2dbackpropinput_transpose_13 = paddle.transpose(x=resfcn256_Conv2d_transpose_12_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_13_conv2d_transpose_conv49_weight = self.resfcn256_Conv2d_transpose_13_conv2d_transpose_conv49_weight
        resfcn256_Conv2d_transpose_13_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_13,
            weight=resfcn256_Conv2d_transpose_13_conv2d_transpose_conv49_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[256, 256])
        resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm = self.bn44(
            resfcn256_Conv2d_transpose_13_conv2d_transpose)
        resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_13_Relu = self.relu44(resfcn256_Conv2d_transpose_13_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_14_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_13_Relu)
        resfcn256_Conv2d_transpose_14_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_14_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_14_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_14_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_14_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_14_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_14_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_14_strided_slice_1, y=resfcn256_Conv2d_transpose_14_mul_y)
        resfcn256_Conv2d_transpose_14_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_14_strided_slice_2, y=resfcn256_Conv2d_transpose_14_mul_1_y)
        resfcn256_Conv2d_transpose_14_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_14_strided_slice, resfcn256_Conv2d_transpose_14_mul,
            resfcn256_Conv2d_transpose_14_mul_1, resfcn256_Conv2d_transpose_14_stack_3
        ])
        resfcn256_Conv2d_transpose_14_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_14_stack, shape=[-1])
        conv2dbackpropinput_transpose_14 = paddle.transpose(x=resfcn256_Conv2d_transpose_13_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_14_conv2d_transpose_conv50_weight = self.resfcn256_Conv2d_transpose_14_conv2d_transpose_conv50_weight
        resfcn256_Conv2d_transpose_14_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_14,
            weight=resfcn256_Conv2d_transpose_14_conv2d_transpose_conv50_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[256, 256])
        resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm = self.bn45(
            resfcn256_Conv2d_transpose_14_conv2d_transpose)
        resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_14_Relu = self.relu45(resfcn256_Conv2d_transpose_14_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_15_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_14_Relu)
        resfcn256_Conv2d_transpose_15_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_15_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_15_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_15_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_15_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_15_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_15_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_15_strided_slice_1, y=resfcn256_Conv2d_transpose_15_mul_y)
        resfcn256_Conv2d_transpose_15_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_15_strided_slice_2, y=resfcn256_Conv2d_transpose_15_mul_1_y)
        resfcn256_Conv2d_transpose_15_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_15_strided_slice, resfcn256_Conv2d_transpose_15_mul,
            resfcn256_Conv2d_transpose_15_mul_1, resfcn256_Conv2d_transpose_15_stack_3
        ])
        resfcn256_Conv2d_transpose_15_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_15_stack, shape=[-1])
        conv2dbackpropinput_transpose_15 = paddle.transpose(x=resfcn256_Conv2d_transpose_14_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_15_conv2d_transpose_conv51_weight = self.resfcn256_Conv2d_transpose_15_conv2d_transpose_conv51_weight
        resfcn256_Conv2d_transpose_15_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_15,
            weight=resfcn256_Conv2d_transpose_15_conv2d_transpose_conv51_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[256, 256])
        resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm = self.bn46(
            resfcn256_Conv2d_transpose_15_conv2d_transpose)
        resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_15_Relu = self.relu46(resfcn256_Conv2d_transpose_15_BatchNorm_FusedBatchNorm)
        resfcn256_Conv2d_transpose_16_Shape = paddle.shape(input=resfcn256_Conv2d_transpose_15_Relu)
        resfcn256_Conv2d_transpose_16_strided_slice = paddle.slice(
            input=resfcn256_Conv2d_transpose_16_Shape, axes=[0], starts=[0], ends=[1])
        resfcn256_Conv2d_transpose_16_strided_slice_1 = paddle.slice(
            input=resfcn256_Conv2d_transpose_16_Shape, axes=[0], starts=[1], ends=[2])
        resfcn256_Conv2d_transpose_16_strided_slice_2 = paddle.slice(
            input=resfcn256_Conv2d_transpose_16_Shape, axes=[0], starts=[2], ends=[3])
        resfcn256_Conv2d_transpose_16_mul = paddle.multiply(
            x=resfcn256_Conv2d_transpose_16_strided_slice_1, y=resfcn256_Conv2d_transpose_16_mul_y)
        resfcn256_Conv2d_transpose_16_mul_1 = paddle.multiply(
            x=resfcn256_Conv2d_transpose_16_strided_slice_2, y=resfcn256_Conv2d_transpose_16_mul_1_y)
        resfcn256_Conv2d_transpose_16_stack = paddle.stack(x=[
            resfcn256_Conv2d_transpose_16_strided_slice, resfcn256_Conv2d_transpose_16_mul,
            resfcn256_Conv2d_transpose_16_mul_1, resfcn256_Conv2d_transpose_16_stack_3
        ])
        resfcn256_Conv2d_transpose_16_stack = paddle.reshape(x=resfcn256_Conv2d_transpose_16_stack, shape=[-1])
        conv2dbackpropinput_transpose_16 = paddle.transpose(x=resfcn256_Conv2d_transpose_15_Relu, perm=[0, 3, 1, 2])
        resfcn256_Conv2d_transpose_16_conv2d_transpose_conv52_weight = self.resfcn256_Conv2d_transpose_16_conv2d_transpose_conv52_weight
        resfcn256_Conv2d_transpose_16_conv2d_transpose = paddle.nn.functional.conv2d_transpose(
            x=conv2dbackpropinput_transpose_16,
            weight=resfcn256_Conv2d_transpose_16_conv2d_transpose_conv52_weight,
            stride=[1, 1],
            dilation=[1, 1],
            padding='SAME',
            output_size=[256, 256])
        resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm = self.bn47(
            resfcn256_Conv2d_transpose_16_conv2d_transpose)
        resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm = paddle.transpose(
            x=resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm, perm=[0, 2, 3, 1])
        resfcn256_Conv2d_transpose_16_Sigmoid = self.sigmoid0(resfcn256_Conv2d_transpose_16_BatchNorm_FusedBatchNorm)
        return resfcn256_Conv2d_transpose_16_Sigmoid


def main(Placeholder):
    # There are 1 inputs.
    # Placeholder: shape-[-1, 256, 256, 3], type-float32.

    paddle.disable_static()
    params = paddle.load('/work/ToTransferInHub/PRNet-Paddle/pd_model/model.pdparams')
    model = TFModel()
    model.set_dict(params, use_structured_name=False)
    model.eval()
    out = model(Placeholder)
    return out


if __name__ == '__main__':
    tensor = paddle.randn([1, 256, 256, 3])
    print(main(tensor).shape)
