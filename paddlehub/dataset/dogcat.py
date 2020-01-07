#coding:utf-8
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import paddlehub as hub
from paddlehub.dataset.base_cv_dataset import BaseCVDataset


class DogCatDataset(BaseCVDataset):
    def __init__(self):
        dataset_path = os.path.join(hub.common.dir.DATA_HOME, "dog-cat")
        base_path = self._download_dataset(
            dataset_path=dataset_path,
            url="https://bj.bcebos.com/paddlehub-dataset/dog-cat.tar.gz")
        super(DogCatDataset, self).__init__(
            base_path=base_path,
            train_list_file="train_list.txt",
            validate_list_file="validate_list.txt",
            test_list_file="test_list.txt",
            label_list_file="label_list.txt",
            label_list=None)


if __name__ == "__main__":
    ds = DogCatDataset()
    print("first 10 dev")
    for e in ds.get_dev_examples()[:10]:
        print(e)
    print("first 10 train")
    for e in ds.get_train_examples()[:10]:
        print(e)
    print("first 10 test")
    for e in ds.get_test_examples()[:10]:
        print(e)
    print(ds)
