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
from paddlehub.datasets import ESC50

import librosa

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument("--sr", type=int, default=44100, help="Sample rate of inference audio.")
parser.add_argument("--model_type", type=str, default='panns_cnn14', help="Select model to to inference.")
parser.add_argument("--checkpoint",
                    type=str,
                    default='./checkpoint/best_model/model.pdparams',
                    help="Checkpoint of model.")
args = parser.parse_args()

if __name__ == '__main__':
    label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}

    model = hub.Module(name=args.model_type,
                       version='1.0.0',
                       task='sound-cls',
                       num_class=ESC50.num_class,
                       label_map=label_map,
                       load_checkpoint=args.checkpoint)

    data = [librosa.load(args.wav, sr=args.sr)[0]]
    result = model.predict(data, sample_rate=args.sr, batch_size=1, feat_type='mel', use_gpu=True)

    print('File: {} \t Lable: {}'.format(os.path.abspath(args.wav), result[0]))
