'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import paddle
import paddle.nn as nn
BN_moment = 0.1

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape

    channels_per_group = num_channels // groups

    # reshape
    x = x.reshape([batchsize, groups,
               channels_per_group, height, width])

    # transpose
    x = paddle.transpose(x, [0, 2, 1, 3, 4])

    # flatten
    x = x.reshape([batchsize, -1, height, width])

    return x


class CBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2D(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias_attr=False)
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class separableCBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2D(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias_attr=False),
            nn.Conv2D(nIn, nOut,  kernel_size=1, stride=1, bias_attr=False),
        )
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class SqueezeBlock(nn.Layer):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()

        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size),
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, exp_size),
                nn.PReLU(exp_size)
            )

    def forward(self, x):
        batch, channels, height, width = x.shape
        out = paddle.nn.functional.avg_pool2d(x, kernel_size=[height, width]).reshape([batch, -1])
        out = self.dense(out)
        out = out.reshape([batch, channels, 1, 1])

        return out * x

class SEseparableCBR(nn.Layer):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, divide=2.0):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2D(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias_attr=False),
            SqueezeBlock(nIn,divide=divide),
            nn.Conv2D(nIn, nOut,  kernel_size=1, stride=1, bias_attr=False),
        )

        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class BR(nn.Layer):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum= BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Layer):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2D(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias_attr=False)
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum= BN_moment)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Layer):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1,group=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2D(nIn, nOut, (kSize, kSize), stride=stride,
                              padding=(padding, padding), bias_attr=False, groups=group)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class S2block(nn.Layer):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, config):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        kSize = config[0]
        avgsize = config[1]

        self.resolution_down = False
        if avgsize >1:
            self.resolution_down = True
            self.down_res = nn.AvgPool2D(avgsize, avgsize)
            self.up_res = nn.Upsample(mode='bilinear', align_corners=True, align_mode=0, scale_factor=avgsize)
            self.avgsize = avgsize

        padding = int((kSize - 1) / 2 )
        self.conv = nn.Sequential(
                        nn.Conv2D(nIn, nIn, kernel_size=(kSize, kSize), stride=1,
                                  padding=(padding, padding), groups=nIn, bias_attr=False),
                        nn.BatchNorm2D(nIn, epsilon=1e-03, momentum=BN_moment))

        self.act_conv1x1 = nn.Sequential(
            nn.PReLU(nIn),
            nn.Conv2D(nIn, nOut, kernel_size=1, stride=1, bias_attr=False),
        )

        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03, momentum=BN_moment)



    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        if self.resolution_down:
            input = self.down_res(input)
        output = self.conv(input)
        output = self.act_conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.bn(output)


class S2module(nn.Layer):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, config= [[3,1],[5,1]]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()

        group_n = len(config)
        n = int(nOut / group_n)
        n1 = nOut - group_n * n

        self.c1 = C(nIn, n, 1, 1, group=group_n)

        for i in range(group_n):
            var_name = 'd{}'.format(i + 1)
            if i == 0:
                self.__dict__["_sub_layers"][var_name] = S2block(n, n + n1, config[i])
            else:
                self.__dict__["_sub_layers"][var_name] = S2block(n, n,  config[i])

        self.BR = BR(nOut)
        self.add = add
        self.group_n = group_n

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        output1= channel_shuffle(output1, self.group_n)

        for i in range(self.group_n):
            var_name = 'd{}'.format(i + 1)
            result_d = self.__dict__["_sub_layers"][var_name](output1)
            if i == 0:
                combine = result_d
            else:
                combine = paddle.concat([combine, result_d], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.BR(combine)
        return output


class InputProjectionA(nn.Layer):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.LayerList()
        for i in range(0, samplingTimes):
            self.pool.append(nn.AvgPool2D(2, stride=2))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input



class SINet_Encoder(nn.Layer):

    def __init__(self, config,classes=20, p=5, q=3,  chnn=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        dim1 = 16
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 96 + 4 * (chnn - 1)

        self.level1 = CBR(3, 12, 3, 2)

        self.level2_0 = SEseparableCBR(12,dim1, 3,2, divide=1)

        self.level2 = nn.LayerList()
        for i in range(0, p):
            if i ==0:
                self.level2.append(S2module(dim1, dim2, config=config[i], add=False))
            else:
                self.level2.append(S2module(dim2, dim2,config=config[i]))
        self.BR2 = BR(dim2+dim1)

        self.level3_0 =SEseparableCBR(dim2+dim1,dim2, 3,2, divide=2)
        self.level3 = nn.LayerList()
        for i in range(0, q):
            if i==0:
                self.level3.append(S2module(dim2, dim3, config=config[2 + i], add=False))
            else:
                self.level3.append(S2module(dim3, dim3,config=config[2+i]))
        self.BR3 = BR(dim3+dim2)

        self.classifier = C(dim3+dim2, classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output1 = self.level1(input) #8h 8w


        output2_0 = self.level2_0(output1)  # 4h 4w

        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2) # 2h 2w


        output3_0 = self.level3_0(self.BR2(paddle.concat([output2_0, output2],1)))  # h w

        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.BR3(paddle.concat([output3_0, output3], 1))

        classifier = self.classifier(output3_cat)

        return classifier


class SINet(nn.Layer):

    def __init__(self,config, classes=20, p=2, q=3, chnn=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        dim2 = 48 + 4 * (chnn - 1)

        self.encoder = SINet_Encoder(config, classes, p, q, chnn)

        self.up = nn.Upsample(mode='bilinear', align_corners=True, align_mode=0, scale_factor=2)
        self.bn_3 = nn.BatchNorm2D(classes, epsilon=1e-03)

        self.level2_C = CBR(dim2, classes, 1, 1)

        self.bn_2 = nn.BatchNorm2D(classes, epsilon=1e-03)

        self.classifier = nn.Sequential(
        nn.Upsample(mode='bilinear', align_corners=True, align_mode=0, scale_factor=2),
        nn.Conv2D(classes, classes, 3, 1, 1, bias_attr=False))

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output1 = self.encoder.level1(input)  # 8h 8w
        output2_0 = self.encoder.level2_0(output1)  # 4h 4w

        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w

        output3_0 = self.encoder.level3_0(self.encoder.BR2(paddle.concat([output2_0, output2], 1)))  # h w

        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.encoder.BR3(paddle.concat([output3_0, output3], 1))
        Enc_final = self.encoder.classifier(output3_cat) #1/8

        Dnc_stage1 = self.bn_3(self.up(Enc_final))  # 1/4
        stage1_confidence = paddle.max(nn.functional.softmax(Dnc_stage1, 1), axis=1)

        b, c, h, w = Dnc_stage1.shape
        stage1_gate = (1-stage1_confidence).unsqueeze(1).expand([b, c, h, w])

        Dnc_stage2_0 = self.level2_C(output2)  # 2h 2w
        Dnc_stage2 = self.bn_2(self.up(Dnc_stage2_0 * stage1_gate + (Dnc_stage1)))  # 4h 4w

        classifier = self.classifier(Dnc_stage2)

        return classifier