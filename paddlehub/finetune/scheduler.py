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


class Compose(object):
    '''
    '''

    def __init__(self, schedulers):
        self.schedulers = schedulers

    def __call__(self, global_lr, parameters):
        for scheduler in self.schedulers:
            global_lr, parameters = scheduler(global_lr, parameters)
        return global_lr, parameters


class WarmUpLR(object):
    '''
    '''

    def __init__(self, start_step: int, end_step: int, wtype=None):
        self.start_step = start_step
        self.end_step = end_step
        self._curr_step = 0

    def __call__(self, global_lr, parameters):
        if self._curr_step >= self.start_step and self._curr_step <= self.end_step:
            global_lr *= float(self._curr_step - self.start_step) / (self.end_step - self.start_step)
        self._curr_step += 1
        return global_lr, parameters


class DecayLR(object):
    '''
    '''

    def __init__(self, start_step: int, end_step: int, wtype=None):
        self.start_step = start_step
        self.end_step = end_step
        self._curr_step = 0

    def __call__(self, global_lr, parameters):
        if self._curr_step >= self.start_step and self._curr_step <= self.end_step:
            global_lr *= float(self.end_step - self._curr_step) / (self.end_step - self.start_step)
        self._curr_step += 1
        return global_lr, parameters


class SlantedTriangleLR(object):
    '''
    '''

    def __init__(self, global_step: int, warmup_prop: float):
        self.global_step = global_step
        self.warmup_prop = warmup_prop
        dividing_line = int(global_step * warmup_prop)
        self.scheduler = Compose([
            WarmUpLR(start_step=0, end_step=dividing_line),
            DecayLR(start_step=dividing_line, end_step=global_step - 1)
        ])

    def __call__(self, global_lr, parameters):
        return self.scheduler(global_lr, parameters)


class GradualUnfreeze(object):
    pass


class LayeredLR(object):
    pass


class L2SP(object):
    '''
    '''

    def __init__(self, regularization_coeff=1e-3):
        self.regularization_coeff = regularization_coeff

    def __call__(self, global_lr, parameters):
        pass
