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

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import paddle

from paddlehub.module.module import RunModule, runnable, serving
from paddlehub.utils.utils import extract_melspectrogram


class AudioClassifierModule(RunModule):
    """
    The base class for audio classifier models.
    """
    _tasks_supported = [
        'sound-cls',
    ]

    def _batchify(self, data: List[List[float]], sample_rate: int, feat_type: str, batch_size: int):
        examples = []
        for waveform in data:
            if feat_type == 'mel':
                feat = extract_melspectrogram(waveform, sample_rate=sample_rate)
                examples.append(feat)
            else:
                examples.append(waveform)

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield one_batch
                one_batch = []
        if one_batch:
            # The last batch whose size is less than the config batch_size setting.
            yield one_batch

    def training_step(self, batch: List[paddle.Tensor], batch_idx: int):
        if self.task == 'sound-cls':
            predictions, avg_loss, metric = self(feats=batch[0], labels=batch[1])
        else:
            raise NotImplementedError

        self.metric.reset()
        return {'loss': avg_loss, 'metrics': metric}

    def validation_step(self, batch: List[paddle.Tensor], batch_idx: int):
        if self.task == 'sound-cls':
            predictions, avg_loss, metric = self(feats=batch[0], labels=batch[1])
        else:
            raise NotImplementedError

        return {'metrics': metric}

    def predict(self,
                data: List[List[float]],
                sample_rate: int,
                batch_size: int = 1,
                feat_type: str = 'mel',
                use_gpu: bool = False):
        if self.task not in self._tasks_supported \
                and self.task is not None:      # None for getting audioset tags
            raise RuntimeError(f'Unknown task {self.task}, current tasks supported:\n'
                               '1. sound-cls: sound classification;\n'
                               '2. None: audioset tags')

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        batches = self._batchify(data, sample_rate, feat_type, batch_size)
        results = []
        self.eval()
        for batch in batches:
            feats = paddle.to_tensor(batch)
            scores = self(feats)

            for score in scores.numpy():
                result = OrderedDict()
                for i in (-score).argsort():
                    result[self.label_map[i]] = score[i]
                results.append(result)

        return results
