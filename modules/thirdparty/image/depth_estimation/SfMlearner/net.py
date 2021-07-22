import paddle
class Conv1(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=32, kernel_size=(7, 7), stride=2, padding=3, in_channels=3)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=32, kernel_size=(7, 7), padding=3, in_channels=32)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv2(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(5, 5), stride=2, padding=2, in_channels=32)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=64, kernel_size=(5, 5), padding=2, in_channels=64)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv3(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), stride=2, padding=1, in_channels=64)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=1, in_channels=128)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv4(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), stride=2, padding=1, in_channels=128)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=256)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv5(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), stride=2, padding=1, in_channels=256)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv6(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv6, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), stride=2, padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Conv7(paddle.nn.Layer):
    def __init__(self, ):
        super(Conv7, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), stride=2, padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
        self.conv1 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=512)
        self.relu1 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        x3 = self.conv1(x2)
        x4 = self.relu1(x3)
        return x4

class Upconv7(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv7, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=512, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv7(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv7, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=1024)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Upconv6(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv6, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=512, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv6(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv6, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=512, kernel_size=(3, 3), padding=1, in_channels=1024)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Upconv5(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv5, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv5(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv5, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=256, kernel_size=(3, 3), padding=1, in_channels=512)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Upconv4(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv4, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=256)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv4(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=128, kernel_size=(3, 3), padding=1, in_channels=256)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Predict_disp4(paddle.nn.Layer):
    def __init__(self, ):
        super(Predict_disp4, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1, kernel_size=(3, 3), padding=1, in_channels=128)
        self.x2 = paddle.nn.Sigmoid()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x3 = self.x2(x1)
        return x3

class Upconv3(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv3, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=64, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=128)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv3(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=64, kernel_size=(3, 3), padding=1, in_channels=129)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Predict_disp3(paddle.nn.Layer):
    def __init__(self, ):
        super(Predict_disp3, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1, kernel_size=(3, 3), padding=1, in_channels=64)
        self.x2 = paddle.nn.Sigmoid()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x3 = self.x2(x1)
        return x3

class Upconv2(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv2, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=64)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv2(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=32, kernel_size=(3, 3), padding=1, in_channels=65)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Predict_disp2(paddle.nn.Layer):
    def __init__(self, ):
        super(Predict_disp2, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1, kernel_size=(3, 3), padding=1, in_channels=32)
        self.x2 = paddle.nn.Sigmoid()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x3 = self.x2(x1)
        return x3

class Upconv1(paddle.nn.Layer):
    def __init__(self, ):
        super(Upconv1, self).__init__()
        self.conv0 = paddle.nn.Conv2DTranspose(out_channels=16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, in_channels=32)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Iconv1(paddle.nn.Layer):
    def __init__(self, ):
        super(Iconv1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=16, kernel_size=(3, 3), padding=1, in_channels=17)
        self.relu0 = paddle.nn.ReLU()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.relu0(x1)
        return x2

class Predict_disp1(paddle.nn.Layer):
    def __init__(self, ):
        super(Predict_disp1, self).__init__()
        self.conv0 = paddle.nn.Conv2D(out_channels=1, kernel_size=(3, 3), padding=1, in_channels=16)
        self.x2 = paddle.nn.Sigmoid()
    def forward(self, x0):
        x1 = self.conv0(x0)
        x3 = self.x2(x1)
        return x3

class DispNetS(paddle.nn.Layer):
    def __init__(self, ):
        super(DispNetS, self).__init__()
        self.conv10 = Conv1()
        self.conv20 = Conv2()
        self.conv30 = Conv3()
        self.conv40 = Conv4()
        self.conv50 = Conv5()
        self.conv60 = Conv6()
        self.conv70 = Conv7()
        self.upconv70 = Upconv7()
        self.iconv70 = Iconv7()
        self.upconv60 = Upconv6()
        self.iconv60 = Iconv6()
        self.upconv50 = Upconv5()
        self.iconv50 = Iconv5()
        self.upconv40 = Upconv4()
        self.iconv40 = Iconv4()
        self.predict_disp40 = Predict_disp4()
        self.upconv30 = Upconv3()
        self.iconv30 = Iconv3()
        self.predict_disp30 = Predict_disp3()
        self.upconv20 = Upconv2()
        self.iconv20 = Iconv2()
        self.predict_disp20 = Predict_disp2()
        self.upconv10 = Upconv1()
        self.iconv10 = Iconv1()
        self.predict_disp10 = Predict_disp1()
    def forward(self, x0):
        x0 = paddle.to_tensor(data=x0)
        x1 = self.conv10(x0)
        x2 = self.conv20(x1)
        x3 = self.conv30(x2)
        x4 = self.conv40(x3)
        x5 = self.conv50(x4)
        x6 = self.conv60(x5)
        x7 = self.conv70(x6)
        x8 = self.upconv70(x7)
        x9 = [0]
        x10 = [0]
        x11 = [2147483647]
        x12 = [1]
        x13 = paddle.strided_slice(x=x8, axes=x9, starts=x10, ends=x11, strides=x12)
        x14 = [1]
        x15 = [0]
        x16 = [2147483647]
        x17 = [1]
        x18 = paddle.strided_slice(x=x13, axes=x14, starts=x15, ends=x16, strides=x17)
        x19 = [2]
        x20 = [0]
        x21 = [2]
        x22 = [1]
        x23 = paddle.strided_slice(x=x18, axes=x19, starts=x20, ends=x21, strides=x22)
        x24 = [3]
        x25 = [0]
        x26 = [7]
        x27 = [1]
        x28 = paddle.strided_slice(x=x23, axes=x24, starts=x25, ends=x26, strides=x27)
        x29 = [x28, x6]
        x30 = paddle.concat(x=x29, axis=1)
        x31 = self.iconv70(x30)
        x32 = self.upconv60(x31)
        x33 = [0]
        x34 = [0]
        x35 = [2147483647]
        x36 = [1]
        x37 = paddle.strided_slice(x=x32, axes=x33, starts=x34, ends=x35, strides=x36)
        x38 = [1]
        x39 = [0]
        x40 = [2147483647]
        x41 = [1]
        x42 = paddle.strided_slice(x=x37, axes=x38, starts=x39, ends=x40, strides=x41)
        x43 = [2]
        x44 = [0]
        x45 = [4]
        x46 = [1]
        x47 = paddle.strided_slice(x=x42, axes=x43, starts=x44, ends=x45, strides=x46)
        x48 = [3]
        x49 = [0]
        x50 = [13]
        x51 = [1]
        x52 = paddle.strided_slice(x=x47, axes=x48, starts=x49, ends=x50, strides=x51)
        x53 = [x52, x5]
        x54 = paddle.concat(x=x53, axis=1)
        x55 = self.iconv60(x54)
        x56 = self.upconv50(x55)
        x57 = [0]
        x58 = [0]
        x59 = [2147483647]
        x60 = [1]
        x61 = paddle.strided_slice(x=x56, axes=x57, starts=x58, ends=x59, strides=x60)
        x62 = [1]
        x63 = [0]
        x64 = [2147483647]
        x65 = [1]
        x66 = paddle.strided_slice(x=x61, axes=x62, starts=x63, ends=x64, strides=x65)
        x67 = [2]
        x68 = [0]
        x69 = [8]
        x70 = [1]
        x71 = paddle.strided_slice(x=x66, axes=x67, starts=x68, ends=x69, strides=x70)
        x72 = [3]
        x73 = [0]
        x74 = [26]
        x75 = [1]
        x76 = paddle.strided_slice(x=x71, axes=x72, starts=x73, ends=x74, strides=x75)
        x77 = [x76, x4]
        x78 = paddle.concat(x=x77, axis=1)
        x79 = self.iconv50(x78)
        x80 = self.upconv40(x79)
        x81 = [0]
        x82 = [0]
        x83 = [2147483647]
        x84 = [1]
        x85 = paddle.strided_slice(x=x80, axes=x81, starts=x82, ends=x83, strides=x84)
        x86 = [1]
        x87 = [0]
        x88 = [2147483647]
        x89 = [1]
        x90 = paddle.strided_slice(x=x85, axes=x86, starts=x87, ends=x88, strides=x89)
        x91 = [2]
        x92 = [0]
        x93 = [16]
        x94 = [1]
        x95 = paddle.strided_slice(x=x90, axes=x91, starts=x92, ends=x93, strides=x94)
        x96 = [3]
        x97 = [0]
        x98 = [52]
        x99 = [1]
        x100 = paddle.strided_slice(x=x95, axes=x96, starts=x97, ends=x98, strides=x99)
        x101 = [x100, x3]
        x102 = paddle.concat(x=x101, axis=1)
        x103 = self.iconv40(x102)
        x104 = self.predict_disp40(x103)
        x105 = 10
        x106 = x104 * x105
        x107 = 0.01
        x108 = x106 + x107
        x109 = self.upconv30(x103)
        x110 = [0]
        x111 = [0]
        x112 = [2147483647]
        x113 = [1]
        x114 = paddle.strided_slice(x=x109, axes=x110, starts=x111, ends=x112, strides=x113)
        x115 = [1]
        x116 = [0]
        x117 = [2147483647]
        x118 = [1]
        x119 = paddle.strided_slice(x=x114, axes=x115, starts=x116, ends=x117, strides=x118)
        x120 = [2]
        x121 = [0]
        x122 = [32]
        x123 = [1]
        x124 = paddle.strided_slice(x=x119, axes=x120, starts=x121, ends=x122, strides=x123)
        x125 = [3]
        x126 = [0]
        x127 = [104]
        x128 = [1]
        x129 = paddle.strided_slice(x=x124, axes=x125, starts=x126, ends=x127, strides=x128)
        x130 = [2.0, 2.0]
        x131 = paddle.nn.functional.interpolate(x=x108, scale_factor=x130, mode='bilinear')
        x132 = [0]
        x133 = [0]
        x134 = [2147483647]
        x135 = [1]
        x136 = paddle.strided_slice(x=x131, axes=x132, starts=x133, ends=x134, strides=x135)
        x137 = [1]
        x138 = [0]
        x139 = [2147483647]
        x140 = [1]
        x141 = paddle.strided_slice(x=x136, axes=x137, starts=x138, ends=x139, strides=x140)
        x142 = [2]
        x143 = [0]
        x144 = [32]
        x145 = [1]
        x146 = paddle.strided_slice(x=x141, axes=x142, starts=x143, ends=x144, strides=x145)
        x147 = [3]
        x148 = [0]
        x149 = [104]
        x150 = [1]
        x151 = paddle.strided_slice(x=x146, axes=x147, starts=x148, ends=x149, strides=x150)
        x152 = [x129, x2, x151]
        x153 = paddle.concat(x=x152, axis=1)
        x154 = self.iconv30(x153)
        x155 = self.predict_disp30(x154)
        x156 = 10
        x157 = x155 * x156
        x158 = 0.01
        x159 = x157 + x158
        x160 = self.upconv20(x154)
        x161 = [0]
        x162 = [0]
        x163 = [2147483647]
        x164 = [1]
        x165 = paddle.strided_slice(x=x160, axes=x161, starts=x162, ends=x163, strides=x164)
        x166 = [1]
        x167 = [0]
        x168 = [2147483647]
        x169 = [1]
        x170 = paddle.strided_slice(x=x165, axes=x166, starts=x167, ends=x168, strides=x169)
        x171 = [2]
        x172 = [0]
        x173 = [64]
        x174 = [1]
        x175 = paddle.strided_slice(x=x170, axes=x171, starts=x172, ends=x173, strides=x174)
        x176 = [3]
        x177 = [0]
        x178 = [208]
        x179 = [1]
        x180 = paddle.strided_slice(x=x175, axes=x176, starts=x177, ends=x178, strides=x179)
        x181 = [2.0, 2.0]
        x182 = paddle.nn.functional.interpolate(x=x159, scale_factor=x181, mode='bilinear')
        x183 = [0]
        x184 = [0]
        x185 = [2147483647]
        x186 = [1]
        x187 = paddle.strided_slice(x=x182, axes=x183, starts=x184, ends=x185, strides=x186)
        x188 = [1]
        x189 = [0]
        x190 = [2147483647]
        x191 = [1]
        x192 = paddle.strided_slice(x=x187, axes=x188, starts=x189, ends=x190, strides=x191)
        x193 = [2]
        x194 = [0]
        x195 = [64]
        x196 = [1]
        x197 = paddle.strided_slice(x=x192, axes=x193, starts=x194, ends=x195, strides=x196)
        x198 = [3]
        x199 = [0]
        x200 = [208]
        x201 = [1]
        x202 = paddle.strided_slice(x=x197, axes=x198, starts=x199, ends=x200, strides=x201)
        x203 = [x180, x1, x202]
        x204 = paddle.concat(x=x203, axis=1)
        x205 = self.iconv20(x204)
        x206 = self.predict_disp20(x205)
        x207 = 10
        x208 = x206 * x207
        x209 = 0.01
        x210 = x208 + x209
        x211 = self.upconv10(x205)
        x212 = [0]
        x213 = [0]
        x214 = [2147483647]
        x215 = [1]
        x216 = paddle.strided_slice(x=x211, axes=x212, starts=x213, ends=x214, strides=x215)
        x217 = [1]
        x218 = [0]
        x219 = [2147483647]
        x220 = [1]
        x221 = paddle.strided_slice(x=x216, axes=x217, starts=x218, ends=x219, strides=x220)
        x222 = [2]
        x223 = [0]
        x224 = [128]
        x225 = [1]
        x226 = paddle.strided_slice(x=x221, axes=x222, starts=x223, ends=x224, strides=x225)
        x227 = [3]
        x228 = [0]
        x229 = [416]
        x230 = [1]
        x231 = paddle.strided_slice(x=x226, axes=x227, starts=x228, ends=x229, strides=x230)
        x232 = [2.0, 2.0]
        x233 = paddle.nn.functional.interpolate(x=x210, scale_factor=x232, mode='bilinear')
        x234 = [0]
        x235 = [0]
        x236 = [2147483647]
        x237 = [1]
        x238 = paddle.strided_slice(x=x233, axes=x234, starts=x235, ends=x236, strides=x237)
        x239 = [1]
        x240 = [0]
        x241 = [2147483647]
        x242 = [1]
        x243 = paddle.strided_slice(x=x238, axes=x239, starts=x240, ends=x241, strides=x242)
        x244 = [2]
        x245 = [0]
        x246 = [128]
        x247 = [1]
        x248 = paddle.strided_slice(x=x243, axes=x244, starts=x245, ends=x246, strides=x247)
        x249 = [3]
        x250 = [0]
        x251 = [416]
        x252 = [1]
        x253 = paddle.strided_slice(x=x248, axes=x249, starts=x250, ends=x251, strides=x252)
        x254 = [x231, x253]
        x255 = paddle.concat(x=x254, axis=1)
        x256 = self.iconv10(x255)
        x257 = self.predict_disp10(x256)
        x258 = 10
        x259 = x257 * x258
        x260 = 0.01
        x261 = x259 + x260
        x262 = (x261, x210, x159, x108)
        return x262

def main(x0):
    # There are 1 inputs.
    # x0: shape-[1, 3, 128, 416], type-float32.
    paddle.disable_static()
    params = paddle.load('pd_model_trace/model.pdparams')
    model = DispNetS()
    model.set_dict(params)
    model.eval()
    out = model(x0)
    return out