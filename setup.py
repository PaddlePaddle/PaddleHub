#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform

from setuptools import find_packages
from setuptools import setup
from paddlehub.version import hub_version


def python_version():
    return [int(v) for v in platform.python_version().split(".")]


max_version, mid_version, min_version = python_version()

REQUIRED_PACKAGES = [
    'six >= 1.10.0', 'protobuf >= 3.6.0', 'pyyaml', 'Pillow', 'requests',
    'tb-paddle', 'tensorboard >= 1.15', 'cma == 2.7.0', 'flask >= 1.1.0',
    'sentencepiece', 'nltk'
]

if max_version < 3:
    REQUIRED_PACKAGES += ["numpy < 1.17.0", "pandas < 0.25.0"]
else:
    REQUIRED_PACKAGES += ["numpy", "pandas"]

setup(
    name='paddlehub',
    version=hub_version.replace('-', ''),
    description=
    ('A toolkit for managing pretrained models of PaddlePaddle and helping user getting started with transfer learning more efficiently.'
     ),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleHub',
    author='PaddlePaddle Author',
    author_email='paddle-dev@baidu.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        'paddlehub/serving/templates': [
            'paddlehub/serving/templates/serving_config.json',
            'paddlehub/serving/templates/main.html'
        ]
    },
    include_package_data=True,
    data_files=[('paddlehub/serving/templates', [
        'paddlehub/serving/templates/serving_config.json',
        'paddlehub/serving/templates/main.html'
    ])],
    include_data_files=True,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=('paddlehub paddlepaddle fine-tune transfer-learning'),
    entry_points={'console_scripts': ['hub=paddlehub.commands.hub:main']})
