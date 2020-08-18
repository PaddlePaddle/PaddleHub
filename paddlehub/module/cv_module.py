# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable

from paddlehub.module.module import serving, RunModule
from paddlehub.utils.utils import base64_to_cv2


class ImageServing(object):
    @serving
    def serving_method(self, images, **kwargs):
        """Run as a service."""
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images=images_decode, **kwargs)
        return results


class ImageClassifierModule(RunModule, ImageServing):
    def training_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = paddle.unsqueeze(batch[1], axes=-1)

        preds = self(images)
        loss, _ = fluid.layers.softmax_with_cross_entropy(preds, labels, return_softmax=True, axis=1)
        loss = fluid.layers.mean(loss)
        acc = fluid.layers.accuracy(preds, labels)
        return {'loss': loss, 'metrics': {'acc': acc}}

    def predict(self, images, top_k=1):
        images = self.transforms(images)
        if len(images.shape) == 3:
            images = images[np.newaxis, :]
        preds = self(to_variable(images))
        preds = fluid.layers.softmax(preds, axis=1).numpy()
        pred_idxs = np.argsort(preds)[::-1][:, :top_k]
        res = []
        for i, pred in enumerate(pred_idxs):
            res_dict = {}
            for k in pred:
                class_name = self.labels[int(k)]
                res_dict[class_name] = preds[i][k]
            res.append(res_dict)
        return res

    def is_better_score(self, old_score, new_score):
        return old_score['acc'] < new_score['acc']
