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

import csv
import io
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle

from paddlehub.utils.utils import extract_melspectrogram


class InputExample(object):
    """
    Input example of one audio sample.
    """

    def __init__(self, guid: int, source: Union[list, str], label: Optional[str] = None):
        self.guid = guid
        self.source = source
        self.label = label


class BaseAudioDataset(object):
    """
    Base class of speech dataset.
    """

    def __init__(self, base_path: str, data_file: str, mode: Optional[str] = "train"):
        self.data_file = os.path.join(base_path, data_file)
        self.mode = mode

    def _read_file(self, input_file: str):
        raise NotImplementedError


class AudioClassificationDataset(BaseAudioDataset, paddle.io.Dataset):
    """
    Base class of audio classification dataset.
    """
    _supported_features = ['raw', 'mel']

    def __init__(self,
                 base_path: str,
                 data_file: str,
                 file_type: str = 'npz',
                 mode: str = 'train',
                 feat_type: str = 'mel',
                 feat_cfg: dict = None):
        super(AudioClassificationDataset, self).__init__(base_path=base_path, mode=mode, data_file=data_file)

        self.file_type = file_type
        self.feat_type = feat_type
        self.feat_cfg = feat_cfg

        self.examples = self._read_file(self.data_file)
        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file: str) -> List[InputExample]:
        if not os.path.exists(input_file):
            raise RuntimeError("Data file: {} not found.".format(input_file))

        examples = []
        if self.file_type == 'npz':
            dataset = np.load(os.path.join(self.data_file), allow_pickle=True)
            audio_id = 0
            for waveform, label in zip(dataset['waveforms'], dataset['labels']):
                example = InputExample(guid=audio_id, source=waveform, label=label)
                audio_id += 1
                examples.append(example)
        else:
            raise NotImplementedError(f'Only soppurts npz file type, but got {self.file_type}')

        return examples

    def _convert_examples_to_records(self, examples: List[InputExample]) -> List[dict]:
        records = []

        for example in examples:
            record = {}
            if self.feat_type == 'raw':
                record['feat'] = example.source
            elif self.feat_type == 'mel':
                record['feat'] = extract_melspectrogram(
                    example.source,
                    sample_rate=self.feat_cfg['sample_rate'],
                    window_size=self.feat_cfg['window_size'],
                    hop_size=self.feat_cfg['hop_size'],
                    mel_bins=self.feat_cfg['mel_bins'],
                    fmin=self.feat_cfg['fmin'],
                    fmax=self.feat_cfg['fmax'],
                    window=self.feat_cfg['window'],
                    center=True,
                    pad_mode='reflect',
                    ref=1.0,
                    amin=1e-10,
                    top_db=None)
            else:
                raise RuntimeError(\
                    f"Unknown type of self.feat_type: {self.feat_type}, it must be one in {self._supported_features}")

            record['label'] = example.label
            records.append(record)

        return records

    def __getitem__(self, idx):
        """
        Overload this method when doing extra feature processes or data augmentation.
        """
        record = self.records[idx]
        return np.array(record['feat']), np.array(record['label'], dtype=np.int64)

    def __len__(self):
        return len(self.records)
