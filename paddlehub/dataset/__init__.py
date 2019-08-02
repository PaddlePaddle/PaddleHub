#coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

# NLP Dataset
from .dataset import InputExample, HubDataset
from .chnsenticorp import ChnSentiCorp
from .msra_ner import MSRA_NER
from .nlpcc_dbqa import NLPCC_DBQA
from .lcqmc import LCQMC
from .toxic import Toxic
from .xnli import XNLI
from .glue import GLUE

# CV Dataset
from .dogcat import DogCatDataset as DogCat
from .flowers import FlowersDataset as Flowers
from .stanford_dogs import StanfordDogsDataset as StanfordDogs
from .food101 import Food101Dataset as Food101
from .indoor67 import Indoor67Dataset as Indoor67
