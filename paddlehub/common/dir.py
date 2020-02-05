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
import contextlib
import shutil
import tempfile

# TODO: Change dir.py's filename, this naming rule is not qualified


def gen_user_home():
    if "HUB_HOME" in os.environ:
        home_path = os.environ["HUB_HOME"]
        if os.path.exists(home_path) and os.path.isdir(home_path):
            return home_path
    return os.path.expanduser('~')


def gen_hub_home():
    return os.path.join(gen_user_home(), ".paddlehub")


USER_HOME = gen_user_home()
HUB_HOME = gen_hub_home()
MODULE_HOME = os.path.join(gen_hub_home(), "modules")
CACHE_HOME = os.path.join(gen_hub_home(), "cache")
DATA_HOME = os.path.join(gen_hub_home(), "dataset")
CONF_HOME = os.path.join(gen_hub_home(), "conf")
THIRD_PARTY_HOME = os.path.join(gen_hub_home(), "thirdparty")
TMP_HOME = os.path.join(gen_hub_home(), "tmp")

if not os.path.exists(TMP_HOME):
    os.makedirs(TMP_HOME)

if not os.path.exists(CACHE_HOME):
    os.makedirs(CACHE_HOME)


@contextlib.contextmanager
def tmp_file():
    with tempfile.NamedTemporaryFile(dir=TMP_HOME) as file:
        yield file.name


@contextlib.contextmanager
def tmp_dir():
    try:
        _dir = tempfile.mkdtemp(dir=TMP_HOME)
        yield _dir
    except:
        raise
    finally:
        shutil.rmtree(_dir)
