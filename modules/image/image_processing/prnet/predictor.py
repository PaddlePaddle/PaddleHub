# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from .pd_model.x2paddle_code import TFModel


class PosPrediction():
    def __init__(self, params, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1

        # network type
        self.network = TFModel()
        self.network.set_dict(params, use_structured_name=False)
        self.network.eval()

    def predict(self, image):
        paddle.disable_static()
        image_tensor = paddle.to_tensor(image[np.newaxis, :, :, :], dtype='float32')
        pos = self.network(image_tensor)
        pos = pos.numpy()
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, feed_dict={self.x: images})
        return pos * self.MaxPos
