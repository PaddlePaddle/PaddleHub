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

import platform

from setuptools import find_packages
from setuptools import setup

import paddlehub as hub

with open("requirements.txt") as fin:
    REQUIRED_PACKAGES = fin.read()

setup(
    name='paddlehub',
    version=hub.__version__.replace('-', ''),
    description=
    ('A toolkit for managing pretrained models of PaddlePaddle and helping user getting started with transfer learning more efficiently.'
     ),
    long_description='',
    url='https://github.com/PaddlePaddle/PaddleHub',
    author='PaddlePaddle Author',
    author_email='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={
        'paddlehub/command/tmpl': [
            'paddlehub/command/tmpl/init_py.tmpl', 'paddlehub/command/tmpl/serving_demo.tmpl',
            'paddlehub/command/tmpl/x_model.tmpl'
        ]
    },
    include_package_data=True,
    data_files=[('paddlehub/commands/tmpl', [
        'paddlehub/commands/tmpl/init_py.tmpl', 'paddlehub/commands/tmpl/serving_demo.tmpl',
        'paddlehub/commands/tmpl/x_model.tmpl'
    ])],
    include_data_files=True,
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords=('paddlehub paddlepaddle fine-tune transfer-learning'),
    entry_points={'console_scripts': ['hub=paddlehub.commands.utils:execute']})
