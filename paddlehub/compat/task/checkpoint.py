# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

import os
from typing import Tuple

import paddle

from paddlehub.compat.task import checkpoint_pb2
from paddlehub.utils.log import logger

CKPT_FILE_NAME = 'ckpt.meta'


def load_checkpoint(checkpoint_dir: str, exe: paddle.static.Executor,
                    main_program: paddle.static.Program) -> Tuple[bool, int, int, float]:

    ckpt_meta_path = os.path.join(checkpoint_dir, CKPT_FILE_NAME)
    ckpt = checkpoint_pb2.CheckPoint()
    logger.info('Try loading checkpoint from {}'.format(ckpt_meta_path))
    if os.path.exists(ckpt_meta_path):
        with open(ckpt_meta_path, 'rb') as f:
            ckpt.ParseFromString(f.read())
    current_epoch = 1
    global_step = 0
    best_score = -999

    if ckpt.latest_model_dir:
        paddle.static.load(executor=exe, model_path=ckpt.latest_model_dir, program=main_program)

        # Compatible with older versions without best_score in checkpoint_pb2
        try:
            best_score = ckpt.best_score
        except:
            best_score = -999

        logger.info('PaddleHub model checkpoint loaded. current_epoch={}, '
                    'global_step={}, best_score={:.5f}'.format(ckpt.current_epoch, ckpt.global_step, best_score))

        return True, ckpt.current_epoch, ckpt.global_step, best_score

    logger.info('PaddleHub model checkpoint not found, start from scratch...')

    return False, current_epoch, global_step, best_score
