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

from setuptools import find_packages
from setuptools import setup

__version__ = "0.1.0-dev"

REQUIRED_PACKAGES = [
    'numpy >= 1.12.0',
    'six >= 1.10.0',
    'protobuf >= 3.4.0',
]

setup(
    name='paddle_hub',
    version=__version__.replace('-', ''),
    description=('PaddleHub is a library to foster the publication, '
                 'discovery, and consumption of reusable parts of machine '
                 'learning models.'),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleHub',
    author='PaddlePaddle Author',
    author_email='chenzeyu01@baidu.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
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
    keywords=('paddlepaddle pretrained paddle-hub'),
)
