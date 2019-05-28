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

from paddle.fluid.framework import Variable

from paddlehub.common.utils import to_list


class Signature(object):
    def __init__(self,
                 name,
                 inputs,
                 outputs,
                 feed_names=None,
                 fetch_names=None,
                 for_predict=False):
        inputs = to_list(inputs)
        outputs = to_list(outputs)

        if not feed_names:
            feed_names = [""] * len(inputs)
        feed_names = to_list(feed_names)
        if len(inputs) != len(feed_names):
            raise ValueError(
                "The length of feed_names must be same with inputs")

        if not fetch_names:
            fetch_names = [""] * len(outputs)
        fetch_names = to_list(fetch_names)
        if len(outputs) != len(fetch_names):
            raise ValueError(
                "the length of fetch_names must be same with outputs")

        self.name = name
        for item in inputs:
            if not isinstance(item, Variable):
                raise TypeError(
                    "Item in inputs list shoule be an instance of fluid.framework.Variable"
                )

        for item in outputs:
            if not isinstance(item, Variable):
                raise TypeError(
                    "Item in outputs list shoule be an instance of fluid.framework.Variable"
                )

        self.inputs = inputs
        self.outputs = outputs
        self.feed_names = feed_names
        self.fetch_names = fetch_names
        self.for_predict = for_predict


def create_signature(name="default",
                     inputs=[],
                     outputs=[],
                     feed_names=None,
                     fetch_names=None,
                     for_predict=False):
    return Signature(
        name=name,
        inputs=inputs,
        outputs=outputs,
        feed_names=feed_names,
        fetch_names=fetch_names,
        for_predict=for_predict)
