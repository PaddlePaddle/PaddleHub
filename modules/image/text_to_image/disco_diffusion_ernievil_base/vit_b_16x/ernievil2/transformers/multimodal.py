# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MultiModalModel(nn.Layer):

    def __init__(self, image_model=None, text_model=None, args=None):
        super(MultiModalModel, self).__init__()
        self.visual = image_model
        self.text_model = text_model

    def encode_text(self, input_ids, pos_ids=None):
        pool_out, text_embedding = self.text_model(input_ids, pos_ids=pos_ids)
        return pool_out

    def encode_image(self, img_word):
        img_embedding = self.visual(img_word)
        return img_embedding[:, 0]

    def forward(self, img_word=None, input_ids=None, pos_ids=None):
        img_embedding = self.visual(img_word)
        img_embedding = img_embedding[:, 0]
        pool_out, text_embedding = self.text_model(input_ids, pos_ids=pos_ids)
        return img_embedding, pool_out
