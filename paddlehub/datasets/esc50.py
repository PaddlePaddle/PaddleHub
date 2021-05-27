#   Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
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

from paddlehub.datasets.base_audio_dataset import AudioClassificationDataset
from paddlehub.env import DATA_HOME
from paddlehub.utils.download import download_data


@download_data(url="https://bj.bcebos.com/paddlehub-dataset/esc50.tar.gz")
class ESC50(AudioClassificationDataset):
    sample_rate = 44100
    input_length = int(sample_rate * 5)  # 5s
    num_class = 50  # class num
    label_list = [
        # Animals
        'Dog',
        'Rooster',
        'Pig',
        'Cow',
        'Frog',
        'Cat',
        'Hen',
        'Insects (flying)',
        'Sheep',
        'Crow',
        # Natural soundscapes & water sounds
        'Rain',
        'Sea waves',
        'Crackling fire',
        'Crickets',
        'Chirping birds',
        'Water drops',
        'Wind',
        'Pouring water',
        'Toilet flush',
        'Thunderstorm',
        # Human, non-speech sounds
        'Crying baby',
        'Sneezing',
        'Clapping',
        'Breathing',
        'Coughing',
        'Footsteps',
        'Laughing',
        'Brushing teeth',
        'Snoring',
        'Drinking, sipping',
        # Interior/domestic sounds
        'Door knock',
        'Mouse click',
        'Keyboard typing',
        'Door, wood creaks',
        'Can opening',
        'Washing machine',
        'Vacuum cleaner',
        'Clock alarm',
        'Clock tick',
        'Glass breaking',
        # Exterior/urban noises
        'Helicopter',
        'Chainsaw',
        'Siren',
        'Car horn',
        'Engine',
        'Train',
        'Church bells',
        'Airplane',
        'Fireworks',
        'Hand saw',
    ]

    def __init__(self, mode: str = 'train', feat_type: str = 'mel'):

        base_path = os.path.join(DATA_HOME, "esc50")

        if mode == 'train':
            data_file = 'train.npz'
        else:
            data_file = 'dev.npz'

        feat_cfg = dict(
            sample_rate=self.sample_rate,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            window='hann')

        super().__init__(
            base_path=base_path,
            data_file=data_file,
            file_type='npz',
            mode=mode,
            feat_type=feat_type,
            feat_cfg=feat_cfg)


if __name__ == "__main__":
    train_dataset = ESC50(mode='train')
    dev_dataset = ESC50(mode='dev')
