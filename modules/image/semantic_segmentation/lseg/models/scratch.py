import numpy as np
import paddle
import paddle.nn as nn


class Interpolate(nn.Layer):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class ResidualConvUnit(nn.Layer):
    """Residual convolution module."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2D(features, features, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2D(features, features, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Layer):
    """Feature fusion block."""

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


class ResidualConvUnit_custom(nn.Layer):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2D(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2D(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2D(features)
            self.bn2 = nn.BatchNorm2D(features)

        self.activation = activation

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return out + x


class FeatureFusionBlock_custom(nn.Layer):
    """Feature fusion block."""

    def __init__(
            self,
            features,
            activation=nn.ReLU(),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2D(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


class Scratch(nn.Layer):

    def __init__(self, in_channels=[256, 512, 1024, 1024], out_channels=256):
        super().__init__()
        self.out_c = 512
        self.logit_scale = paddle.to_tensor(np.exp(np.log([1 / 0.07])))
        self.layer1_rn = nn.Conv2D(
            in_channels[0],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            groups=1,
        )
        self.layer2_rn = nn.Conv2D(
            in_channels[1],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            groups=1,
        )
        self.layer3_rn = nn.Conv2D(
            in_channels[2],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            groups=1,
        )
        self.layer4_rn = nn.Conv2D(
            in_channels[3],
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False,
            groups=1,
        )

        self.refinenet1 = FeatureFusionBlock_custom(out_channels, bn=True)
        self.refinenet2 = FeatureFusionBlock_custom(out_channels, bn=True)
        self.refinenet3 = FeatureFusionBlock_custom(out_channels, bn=True)
        self.refinenet4 = FeatureFusionBlock_custom(out_channels, bn=True)

        self.head1 = nn.Conv2D(out_channels, self.out_c, kernel_size=1)

        self.output_conv = nn.Sequential(Interpolate(scale_factor=2, mode="bilinear", align_corners=True))

    def forward(self, layer_1, layer_2, layer_3, layer_4, text_features):

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        image_features = self.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.transpose((0, 2, 3, 1)).reshape((-1, self.out_c))

        # normalized features
        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(axis=-1, keepdim=True)

        logits_per_image = self.logit_scale * image_features @ text_features.t()

        out = logits_per_image.reshape((imshape[0], imshape[2], imshape[3], -1)).transpose((0, 3, 1, 2))

        out = self.output_conv(out)

        return out
