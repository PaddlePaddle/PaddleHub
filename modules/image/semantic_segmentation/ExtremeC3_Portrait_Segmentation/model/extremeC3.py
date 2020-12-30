'''
ExtPortraitSeg
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import paddle
import paddle.nn as nn

basic_0 = 24
basic_1 = 48
basic_2 = 56
basic_3 = 24

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
        # self.conv = nn.Conv2D(nIn, nOut, kSize, stride=stride, padding=padding, bias_attr=False)
        self.conv = nn.Conv2D(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias_attr=False)
        # self.conv1 = nn.Conv2D(nOut, nOut, (1, kSize), stride=1, padding=(0, padding), bias_attr=False)
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03)
        self.act = nn.PReLU(nOut)
        # self.act = nn.ReLU()


    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
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
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03)
        self.act = nn.PReLU(nOut)
        # self.act = nn.ReLU()

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
        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-03)

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

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2D(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias_attr=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output



class C3block(nn.Layer):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv =nn.Sequential(
                nn.Conv2D(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias_attr=False,
                          dilation=d),
                nn.Conv2D(nIn, nOut, kernel_size=1, stride=1, bias_attr=False)
            )
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2D(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias_attr=False),
                nn.BatchNorm2D(nIn),
                nn.PReLU(nIn),
                nn.Conv2D(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias_attr=False),
                nn.BatchNorm2D(nIn),
                nn.Conv2D(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias_attr=False,
                          dilation=d),
                nn.Conv2D(nIn, nOut, kernel_size=1, stride=1, bias_attr=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class Down_advancedC3(nn.Layer):
    def __init__(self, nIn, nOut, ratio=[2,4,8]):
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 3, 2)

        self.d1 = C3block(n, n+n1, 3, 1, ratio[0])
        self.d2 = C3block(n, n, 3, 1, ratio[1])
        self.d3 = C3block(n, n, 3, 1, ratio[2])

        self.bn = nn.BatchNorm2D(nOut, epsilon=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = paddle.concat([d1, d2, d3], 1)

        output = self.bn(combine)
        output = self.act(output)
        return output

class AdvancedC3(nn.Layer):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, ratio=[2,4,8]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut // 3)
        n1 = nOut - 3 * n
        self.c1 = C(nIn, n, 1, 1)

        self.d1 = C3block(n, n + n1, 3, 1, ratio[0])
        self.d2 = C3block(n, n, 3, 1, ratio[1])
        self.d3 = C3block(n, n, 3, 1, ratio[2])
        # self.d4 = Double_CDilated(n, n, 3, 1, 12)
        # self.conv =C(nOut, nOut, 1,1)

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)

        combine = paddle.concat([d1, d2, d3], 1)

        if self.add:
            combine = input + combine
        output = self.bn(combine)
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
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2D(2, stride=2, padding=0))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input

class ExtremeC3NetCoarse(nn.Layer):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()


        self.level1 = CBR(3, basic_0, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(basic_0 + 3)
        self.level2_0 = Down_advancedC3(basic_0 + 3, basic_1, ratio=[1, 2, 3])  # , ratio=[1,2,3]

        self.level2 = nn.LayerList()
        for i in range(0, p):
            self.level2.append(
                AdvancedC3(basic_1, basic_1, ratio=[1, 3, 4]))  # , ratio=[1,3,4]
        self.b2 = BR(basic_1 * 2 + 3)

        self.level3_0 = AdvancedC3(basic_1 * 2 + 3, basic_2, add=False,
                                                            ratio=[1, 3, 5])  # , ratio=[1,3,5]

        self.level3 = nn.LayerList()
        for i in range(0, q):
            self.level3.append(AdvancedC3(basic_2, basic_2))
        self.b3 = BR(basic_2 * 2)


        self.Coarseclassifier = C(basic_2*2, classes, 1, 1)



    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(paddle.concat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(paddle.concat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(paddle.concat([output2_0, output2], 1))

        classifier = self.Coarseclassifier(output2_cat)
        return classifier

class ExtremeC3Net(nn.Layer):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()


        self.encoder = ExtremeC3NetCoarse(classes, p, q)
        # # load the encoder modules
        del self.encoder.Coarseclassifier

        self.upsample = nn.Sequential(
            nn.Conv2D(kernel_size=(1, 1), in_channels=basic_2*2, out_channels=basic_3,bias_attr=False),
            nn.BatchNorm2D(basic_3),
            nn.Upsample(mode='bilinear', align_corners=True, align_mode=0, scale_factor=2)

        )

        self.Fine = nn.Sequential(
            # nn.Conv2D(kernel_size=3, stride=2, padding=1, in_channels=3, out_channels=basic_3,bias_attr=False),
            C(3, basic_3, 3, 2),
            AdvancedC3(basic_3, basic_3, add=True),

            # nn.BatchNorm2D(basic_3, epsilon=1e-03),

        )
        self.classifier = nn.Sequential(
            BR(basic_3),
            nn.Upsample(mode='bilinear', align_corners=True, align_mode=0, scale_factor=2),
            nn.Conv2D(kernel_size=(1, 1), in_channels=basic_3, out_channels=classes, bias_attr=False),
        )

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.encoder.level1(input)
        inp1 = self.encoder.sample1(input)
        inp2 = self.encoder.sample2(input)

        output0_cat = self.encoder.b1(paddle.concat([output0, inp1], 1))
        output1_0 = self.encoder.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.encoder.b2(paddle.concat([output1, output1_0, inp2], 1))

        output2_0 = self.encoder.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.encoder.b3(paddle.concat([output2_0, output2], 1))

        Coarse = self.upsample(output2_cat)
        Fine =  self.Fine(input)
        classifier = self.classifier(Coarse +Fine)
        
        return classifier