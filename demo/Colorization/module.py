# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import paddle.fluid as fluid
import paddle.nn as nn
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, Conv2DTranspose
from paddlehub.module.module import moduleinfo

from Colorization.process.transforms import *
from Colorization.cv_module import ImageColorizeModule


@moduleinfo(
    name="Colorization",
    type="CV/image_editing",
    author="baidu-vis",
    author_email="",
    summary="User_guided_colorization is a image colorization model, this module is trained with ILSVRC2012 dataset.",
    version="1.0.0",
    meta=ImageColorizeModule)
class Userguidedcolorization(fluid.dygraph.Layer):
    def __init__(self, use_tanh=True, classification=True, load_checkpoint=True):
        super(Userguidedcolorization, self).__init__()
        self.input_nc = 4
        self.output_nc = 2
        self.classification = classification

        # Conv1
        model1 = (Conv2D(self.input_nc, 64, 3, 1, 1, act='relu'),
                  Conv2D(64, 64, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(64),)
        # Conv2
        model2 = (Conv2D(64, 128, 3, 1, 1, act='relu'),
                  Conv2D(128, 128, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(128),)
        # Conv3
        model3 = (Conv2D(128, 256, 3, 1, 1, act='relu'),
                  Conv2D(256, 256, 3, 1, 1, act='relu'),
                  Conv2D(256, 256, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(256),)
        # Conv4
        model4 = (Conv2D(256, 512, 3, 1, 1, act='relu'),
                  Conv2D(512, 512, 3, 1, 1, act='relu'),
                  Conv2D(512, 512, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(512),)
        # Conv5
        model5 = (Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  fluid.BatchNorm(512),)
        # Conv6
        model6 = (Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  Conv2D(512, 512, 3, 1, 2, 2, act='relu'),
                  fluid.BatchNorm(512),)
        # Conv7
        model7 = (Conv2D(512, 512, 3, 1, 1, act='relu'),
                  Conv2D(512, 512, 3, 1, 1, act='relu'),
                  Conv2D(512, 512, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(512),)
        # Conv8
        model8up = (Conv2DTranspose(512, 256, 4, None, 1, 2),)
        model3short8 = (Conv2D(256, 256, 3, 1, 1),)
        model8 = (Conv2D(256, 256, 3, 1, 1, act='relu'),
                  Conv2D(256, 256, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(256),)
        # Conv9
        model9up = (Conv2DTranspose(256, 128, 4, None, 1, 2),)
        model2short9 = (Conv2D(128, 128, 3, 1, 1, ),)
        model9 = (Conv2D(128, 128, 3, 1, 1, act='relu'),
                  fluid.BatchNorm(128),)
        # Conv10
        model10up = (Conv2DTranspose(128, 128, 4, None, 1, 2),)
        model1short10 = (Conv2D(64, 128, 3, 1, 1),)
        model10 = (Conv2D(128, 128, 3, 1, 1),)
        model_class = (Conv2D(256, 529, 1),)
        model_out = (Conv2D(128, 2, 1, 1, 0, 1),)

        self.model1 = fluid.dygraph.Sequential(*model1)
        self.model2 = fluid.dygraph.Sequential(*model2)
        self.model3 = fluid.dygraph.Sequential(*model3)
        self.model4 = fluid.dygraph.Sequential(*model4)
        self.model5 = fluid.dygraph.Sequential(*model5)
        self.model6 = fluid.dygraph.Sequential(*model6)
        self.model7 = fluid.dygraph.Sequential(*model7)
        self.model8up = fluid.dygraph.Sequential(*model8up)
        self.model3short8 = fluid.dygraph.Sequential(*model3short8)
        self.model8 = fluid.dygraph.Sequential(*model8)
        self.model9up = fluid.dygraph.Sequential(*model9up)
        self.model2short9 = fluid.dygraph.Sequential(*model2short9)
        self.model9 = fluid.dygraph.Sequential(*model9)
        self.model10up = fluid.dygraph.Sequential(*model10up)
        self.model1short10 = fluid.dygraph.Sequential(*model1short10)
        self.model10 = fluid.dygraph.Sequential(*model10)
        self.model_class = fluid.dygraph.Sequential(*model_class)
        self.model_out = fluid.dygraph.Sequential(*model_out)
        self.upsample4 = fluid.dygraph.Sequential(*(nn.UpSample(scale=4, resample='NEAREST'),))
        self.use_tanh = use_tanh
        if load_checkpoint:
            model_dict = fluid.dygraph.load_dygraph('model')[0]
            self.set_dict(model_dict)
            print("load pretrained model success")

    def transforms(self, images, is_train=True):
        if is_train:
            transform = Compose([Resize((256, 256), interp="RANDOM"),
                                 RandomPaddingCrop(crop_size=176),
                                 ColorConvert(mode='RGB2LAB'),
                                 ColorizePreprocess(ab_thresh=0, is_train=is_train)])
        else:
            transform = Compose([Resize((256, 256), interp="RANDOM"),
                                 ColorConvert(mode='RGB2LAB'),
                                 ColorizePreprocess(ab_thresh=0, is_train=is_train)])
        return transform(images)

    def forward(self, input_A, input_B, mask_B, real_b = None, real_B_enc=None):
        conv1_2 = self.model1(fluid.layers.concat([input_A, input_B, mask_B], axis=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = fluid.layers.relu(self.model8up(conv7_3) + self.model3short8(conv3_3))
        conv8_3 = self.model8(conv8_up)
        if self.classification:
            out_class = self.model_class(conv8_3)
            conv9_up = fluid.layers.relu(self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach()))
            conv9_3 = self.model9(conv9_up)
            conv10_up = fluid.layers.relu(self.model10up(conv9_3) + self.model1short10(conv1_2.detach()))
            conv10_2 = fluid.layers.leaky_relu(self.model10(conv10_up), alpha=0.2)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())
            conv9_up = fluid.layers.relu(self.model9up(conv8_3) + self.model2short9(conv2_2))
            conv9_3 = self.model9(conv9_up)
            conv10_up = fluid.layers.relu(self.model10up(conv9_3) + self.model1short10(conv1_2))
            conv10_2 = fluid.layers.leaky_relu(self.model10(conv10_up), alpha=0.2)
            out_reg = self.model_out(conv10_2)

        if self.use_tanh:
            out_reg = fluid.layers.tanh(out_reg)

        return out_class, out_reg











