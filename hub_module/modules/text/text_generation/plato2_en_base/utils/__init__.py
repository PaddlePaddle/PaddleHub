#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Utils."""

from itertools import chain
import os
import time

import numpy as np
import paddle.fluid as fluid


def to_lodtensor(data, place):
    """Convert data to LoDTensor."""
    if place is None:
        return data
    lengths = []
    while isinstance(data[0], list):
        lengths.append(list(map(len, data)))
        data = [x for xs in data for x in xs]
    if isinstance(data[0], float):
        data = np.array(data, dtype="float32")
    else:
        data = np.array(data, dtype="int64")
    data_tensor = fluid.LoDTensor()
    data_tensor.set(data, place)
    data_tensor.set_recursive_sequence_lengths(lengths)
    return data_tensor


def pad_batch_data(insts, pad_id=0):
    """Pad the instances to the max sequence length in batch. """
    max_len = max(map(len, insts))
    inst_data = np.array([list(inst) + [pad_id] * (max_len - len(inst)) for inst in insts])
    return inst_data.astype("int64").reshape([-1, max_len, 1])


def convert_lodtensor_to_list(tensor):
    data = np.array(tensor)
    recursive_sequence_lengths = tensor.recursive_sequence_lengths()
    recursive_sequence_lengths.reverse()
    for i, lengths in enumerate(recursive_sequence_lengths):
        shift = 0
        new_data = []
        for j, l in enumerate(lengths):
            new_data.append(data[shift:shift + l])
            shift += l
        data = new_data
    return data


def concatenate_lodtensors(tensors, place):
    """Concatenate LoD tensors."""
    data = []
    recursive_sequence_lengths = []
    for tensor in tensors:
        data.append(np.array(tensor))
        recursive_sequence_lengths.append(tensor.recursive_sequence_lengths())
    data = np.concatenate(data, axis=0)
    recursive_sequence_lengths = [sum(lens, []) for lens in zip(*recursive_sequence_lengths)]
    data_tensor = fluid.LoDTensor()
    data_tensor.set(data, place)
    data_tensor.set_recursive_sequence_lengths(recursive_sequence_lengths)
    assert data_tensor.has_valid_recursive_sequence_lengths()
    return data_tensor


def repeat_array_or_tensor(array_or_tensor, place, times):
    """Repeate numpy array or LoD tensor."""
    if isinstance(array_or_tensor, fluid.LoDTensor):
        data = [np.array(array_or_tensor)] * times
        recursive_sequence_lengths = [array_or_tensor.recursive_sequence_lengths()] * times
        data = np.concatenate(data, axis=0)
        recursive_sequence_lengths = [sum(lens, []) for lens in zip(*recursive_sequence_lengths)]
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, place)
        data_tensor.set_recursive_sequence_lengths(recursive_sequence_lengths)
        assert data_tensor.has_valid_recursive_sequence_lengths()
        return data_tensor
    elif isinstance(array_or_tensor, list):
        return list(chain(*([array_or_tensor] * times)))
    else:
        return np.concatenate([array_or_tensor] * times, axis=0)


def slice_array_or_tensor(array_or_tensor, place, begin, end):
    """Repeate numpy array or LoD tensor."""
    if isinstance(array_or_tensor, fluid.LoDTensor):
        data = convert_lodtensor_to_list(array_or_tensor)
        data = data[begin:end]
        return to_lodtensor(data, place)
    else:
        return array_or_tensor[begin:end]


def init_checkpoint(exe, init_checkpoint_path, main_program):
    """Initialize from checkpoint."""
    assert os.path.exists(
        init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path
    
    def existed_persitables(var):
        """Whether var is a persistables."""
        if not fluid.io.is_persistable(var):
            return False
        return os.path.exists(os.path.join(init_checkpoint_path, var.name))

    fluid.io.load_vars(
        exe,
        init_checkpoint_path,
        main_program=main_program,
        predicate=existed_persitables)
    print(f"Load model from {init_checkpoint_path}")


def init_pretraining_params(exe,
                            pretraining_params_path,
                            main_program):
    """Only initialize parameters."""
    assert os.path.exists(pretraining_params_path
                          ), "[%s] cann't be found." % pretraining_params_path

    def existed_params(var):
        """Whether var is a parameter."""
        if not isinstance(var, fluid.framework.Parameter):
            return False
        return os.path.exists(os.path.join(pretraining_params_path, var.name))

    fluid.io.load_vars(
        exe,
        pretraining_params_path,
        main_program=main_program,
        predicate=existed_params)
    print(f"Load pretraining parameters from {pretraining_params_path}.")

    return


class Timer(object):

    def __init__(self):
        self._pass_time = 0
        self._start_time = None
        return

    def start(self):
        self._start_time = time.time()

    def pause(self):
        self._pass_time += time.time() - self._start_time
        self._start_time = None

    def reset(self):
        self._pass_time = 0

    @property
    def pass_time(self):
        if self._start_time is None:
            return self._pass_time
        else:
            return self._pass_time + time.time() - self._start_time


ERROR_MESSAGE="\nYou can not set use_cuda = True in the model because you are using paddlepaddle-cpu.\n \
    Please: 1. Install paddlepaddle-gpu to run your models on GPU or 2. Set use_cuda = False to run models on CPU.\n"
def check_cuda(use_cuda, err=ERROR_MESSAGE):
    """Check CUDA."""
    try:
        if use_cuda and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass
