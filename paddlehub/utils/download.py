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

import os

from paddlehub.env import DATA_HOME
from paddle.utils.download import get_path_from_url


def download_data(url):
    save_name = os.path.basename(url).split('.')[0]
    output_path = os.path.join(DATA_HOME, save_name)

    if not os.path.exists(output_path):
        get_path_from_url(url, DATA_HOME)

    def _wrapper(Dataset):
        return Dataset

    return _wrapper
