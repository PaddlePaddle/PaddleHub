#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle_hub.finetune import checkpoint_pb2


def load_checkpoint(checkpoint_path):
    ckpt = checkpoint_pb2.CheckPoint()
    with open(checkpoint_path, "rb") as file:
        ckpt.ParseFromString(file.read())
    return ckpt.last_epoch, ckpt.last_step, ckpt.last_model_dir


def save_checkpoint(checkpoint_path, last_epoch, last_step, last_model_dir):
    ckpt = checkpoint_pb2.CheckPoint()
    ckpt.last_epoch = last_epoch
    ckpt.last_step = last_step
    ckpt.last_model_dir = last_model_dir
    with open(checkpoint_path, "wb") as file:
        file.write(ckpt.SerializeToString())
