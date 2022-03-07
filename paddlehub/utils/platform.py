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

import sys
import platform


def get_platform() -> str:
    return platform.platform()


def is_windows() -> bool:
    return get_platform().lower().startswith("windows")


def get_platform_info() -> dict:
    return {
        'python_version': '.'.join(map(str, sys.version_info[0:3])),
        'platform_version': platform.version(),
        'platform_system': platform.system(),
        'platform_architecture': platform.architecture(),
        'platform_type': platform.platform()
    }
