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

import os
import ast
import argparse

import paddlehub as hub
from paddlehub.env import MODULE_HOME

import librosa
import numpy as np

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument("--sr", type=int, default=32000, help="Sample rate of inference audio.")
parser.add_argument("--model_type", type=str, default='panns_cnn14', help="Select model to to inference.")
parser.add_argument("--topk", type=int, default=10, help="Show top k results of audioset labels.")
args = parser.parse_args()


def show_topk(k, label_map, file, result):
    result = np.asarray(result)
    topk_idx = (-result).argsort()[:k]
    msg = f'[{file}]\n'
    for idx in topk_idx:
        label, score = label_map[idx], result[idx]
        msg += f'{label}: {score}\n'
    print(msg)


if __name__ == '__main__':
    model = hub.Module(name=args.model_type, version='1.0.0', task=None)

    label_file = os.path.join(MODULE_HOME, args.model_type, 'audioset_labels.txt')
    label_map = {}
    with open(label_file, 'r') as f:
        for i, l in enumerate(f.readlines()):
            label_map[i] = l.strip()

    data = [librosa.load(args.wav, sr=args.sr)[0]]  # (t, num_mel_bins)
    result = model.predict(data, sample_rate=args.sr, batch_size=1, feat_type='mel', use_gpu=True)

    show_topk(args.topk, label_map, os.path.abspath(args.wav), result[0])
