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

import os

import paddle
import paddle.nn as nn
from paddle.nn import Conv2d, ConvTranspose2d
from paddlehub.module.module import moduleinfo
import paddlehub.process.transforms as T
from paddlehub.module.cv_module import ImageColorizeModule
from user_guided_colorization.data_feed import ColorizePreprocess


@moduleinfo(
    name="user_guided_colorization",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="User_guided_colorization is a image colorization model, this module is trained with ILSVRC2012 dataset.",
    version="1.0.0",
    meta=ImageColorizeModule)
class UserGuidedColorization(nn.Layer):
    """
    Userguidedcolorization, see https://github.com/haoyuying/colorization-pytorch

    Args:
        use_tanh (bool): Whether to use tanh as final activation function.
        classification (bool): Whether to switch classification branch for optimization.
        load_checkpoint (str): Pretrained checkpoint path.

    """
    def __init__(self, use_tanh: bool = True, classification: bool = True, load_checkpoint: str = None):
        super(UserGuidedColorization, self).__init__()
        self.input_nc = 4
        self.output_nc = 2
        self.classification = classification
        # Conv1
        model1 = (
            Conv2d(self.input_nc, 64, 3, 1, 1),
            nn.ReLU(),
            Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(64),
        )

        # Conv2
        model2 = (
            Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(128),
        )

        # Conv3
        model3 = (
            Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(256),
        )

        # Conv4
        model4 = (
            Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(512),
        )

        # Conv5
        model5 = (
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            nn.BatchNorm(512),
        )

        # Conv6
        model6 = (
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 2, 2),
            nn.ReLU(),
            nn.BatchNorm(512),
        )

        # Conv7
        model7 = (
            Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(512),
        )

        # Conv8
        model8up = (ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), )
        model3short8 = (Conv2d(256, 256, 3, 1, 1), )
        model8 = (
            nn.ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm(256),
        )

        # Conv9
        model9up = (ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), )
        model2short9 = (Conv2d(
            128,
            128,
            3,
            1,
            1,
        ), )
        model9 = (nn.ReLU(), Conv2d(128, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm(128))

        # Conv10
        model10up = (ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), )
        model1short10 = (Conv2d(64, 128, 3, 1, 1), )
        model10 = (nn.ReLU(), Conv2d(128, 128, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2))
        model_class = (Conv2d(256, 529, 1), )

        if use_tanh:
            model_out = (Conv2d(128, 2, 1, 1, 0, 1), nn.Tanh())
        else:
            model_out = (Conv2d(128, 2, 1, 1, 0, 1), )

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)
        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        if load_checkpoint is not None:
            model_dict = paddle.load(load_checkpoint)[0]
            self.set_dict(model_dict)
            print("load custom checkpoint success")
        else:
            checkpoint = os.path.join(self.directory, 'user_guided.pdparams')
            if not os.path.exists(checkpoint):
                os.system('wget https://paddlehub.bj.bcebos.com/dygraph/image_colorization/user_guided.pdparams -O ' +
                          checkpoint)
            model_dict = paddle.load(checkpoint)[0]
            self.set_dict(model_dict)
            print("load pretrained checkpoint success")

    def transforms(self, images: str, is_train: bool = True) -> callable:
        if is_train:
            transform = T.Compose(
                [T.Resize((256, 256), interp='NEAREST'),
                 T.RandomPaddingCrop(crop_size=176),
                 T.RGB2LAB()],
                stay_rgb=True,
                is_permute=False)
        else:
            transform = T.Compose([T.Resize(
                (256, 256), interp='NEAREST'), T.RGB2LAB()],
                                  stay_rgb=True,
                                  is_permute=False)
        return transform(images)

    def preprocess(self, inputs: paddle.Tensor, ab_thresh: float = 0., prob: float = 0.):
        self.preprocess = ColorizePreprocess(ab_thresh=ab_thresh, p=prob)
        return self.preprocess(inputs)

    def forward(self,
                input_A: paddle.Tensor,
                input_B: paddle.Tensor,
                mask_B: paddle.Tensor,
                real_b: paddle.Tensor = None,
                real_B_enc: paddle.Tensor = None) -> paddle.Tensor:
        conv1_2 = self.model1(paddle.concat([input_A, input_B, mask_B], axis=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if self.classification:
            out_class = self.model_class(conv8_3)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())
            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        return out_class, out_reg


if __name__ == "__main__":
    place = paddle.CUDAPlace(0)
    paddle.disable_static()
    model = UserGuidedColorization()
    model.eval()
