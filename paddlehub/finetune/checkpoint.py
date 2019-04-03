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

import os

import paddle.fluid as fluid

from paddlehub.finetune import checkpoint_pb2
from paddlehub.common.logger import logger

CKPT_FILE_NAME = "ckpt.meta"


def load_checkpoint(checkpoint_dir, exe):
    ckpt_meta_path = os.path.join(checkpoint_dir, CKPT_FILE_NAME)
    logger.info("Try loading checkpoint from {}".format(ckpt_meta_path))
    if os.path.exists(ckpt_meta_path):
        ckpt = checkpoint_pb2.CheckPoint()
        with open(ckpt_meta_path, "rb") as f:
            ckpt.ParseFromString(f.read())

        fluid.io.load_persistables(exe, ckpt.latest_model_dir)

        logger.info("PaddleHub model checkpoint loaded. current_epoch={}, "
                    "global_step={}".format(ckpt.current_epoch,
                                            ckpt.global_step))
        return ckpt.current_epoch, ckpt.global_step
    else:
        current_epoch = 1
        global_step = 0
        latest_model_dir = None
        logger.info(
            "PaddleHub model checkpoint not found, start training from scratch..."
        )
        exe.run(fluid.default_startup_program())

        return current_epoch, global_step


def save_checkpoint(checkpoint_dir, current_epoch, global_step, exe):
    ckpt_meta_path = os.path.join(checkpoint_dir, CKPT_FILE_NAME)
    ckpt = checkpoint_pb2.CheckPoint()

    model_saved_dir = os.path.join(checkpoint_dir, "step_%d" % global_step)
    logger.info("Saving model checkpoint to {}".format(model_saved_dir))
    fluid.io.save_persistables(exe, dirname=model_saved_dir)

    ckpt.current_epoch = current_epoch
    ckpt.global_step = global_step
    ckpt.latest_model_dir = model_saved_dir
    with open(ckpt_meta_path, "wb") as f:
        f.write(ckpt.SerializeToString())
