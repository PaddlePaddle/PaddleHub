# -*- coding: UTF-8 -*-
################################################################################
#
#   Copyright (c) 2020  Baidu, Inc. All Rights Reserved.
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
#################################################################################
"""本文件自定义的网络工具函数"""

import numpy as np

from paddle.fluid.core_avx import VarDesc
from paddle.fluid import layers


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """将长度不同sequences(np.ndarray)填充为固定长度的矩阵"""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def pad_sequence_paddle(sequences, padding_value=0):
    """将长度不同sequences(variable)填充为固定长度的矩阵"""
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    out_tensor = []
    for tensor in sequences:
        length = tensor.shape[0]
        pad_tensor = layers.concat((tensor,
                                    layers.fill_constant(
                                        (max_len - length, *trailing_dims),
                                        dtype=tensor.dtype,
                                        value=padding_value)))
        out_tensor.append(pad_tensor)
    out_tensor = layers.stack(out_tensor)
    return out_tensor


def fill_diagonal(x, value, offset=0, dim1=0, dim2=1):
    """将dim1，dim2这两个维度偏移量为offset的对角线置为value"""
    strides = x.strides
    shape = x.shape
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
    assert 0 <= dim1 < dim2 <= 2
    assert len(x.shape) == 3
    assert shape[dim1] == shape[dim2]

    dim_sum = dim1 + dim2
    dim3 = 3 - dim_sum
    if offset >= 0:
        diagonal = np.lib.stride_tricks.as_strided(
            x[:, offset:] if dim_sum == 1 else x[:, :, offset:],
            shape=(shape[dim3], shape[dim1] - offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))
    else:
        diagonal = np.lib.stride_tricks.as_strided(
            x[-offset:, :] if dim_sum in [1, 2] else x[:, -offset:],
            shape=(shape[dim3], shape[dim1] + offset),
            strides=(strides[dim3], strides[dim1] + strides[dim2]))

    diagonal[...] = value
    return x


def backtrack(p_i, p_c, heads, i, j, complete):
    """回溯eisner算法记录的位置矩阵，生成依存句法树"""
    if i == j:
        return
    if complete:
        r = p_c[i, j]
        backtrack(p_i, p_c, heads, i, r, False)
        backtrack(p_i, p_c, heads, r, j, True)
    else:
        r, heads[j] = p_i[i, j], i
        i, j = sorted((i, j))
        backtrack(p_i, p_c, heads, i, r, True)
        backtrack(p_i, p_c, heads, j, r + 1, True)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r'''Returns a diagonal stripe of the tensor.

    Args:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = np.arange(25).reshape(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    '''
    x = np.ascontiguousarray(x)
    strides = x.strides
    m = strides[0] + strides[1]
    k = strides[1] if dim == 1 else strides[0]
    return np.lib.stride_tricks.as_strided(
        x[offset[0]:, offset[1]:],
        shape=(n, w, *x.shape[2:]),
        strides=(m, k, *strides[2:]))


def masked_select(input, mask):
    """将mask中True的索引选择input的值

    Arags：
        input: 输入
        mask: mask矩阵

    Returns:
        经mask选择过的input

    >>> input
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> mask
    [
        [True, True, False],
        [True, False, False]
    ]
    >>> masked_select(input, mask)
    [1, 2, 4]
    """
    select = layers.where(mask)
    output = layers.gather_nd(input, select)
    return output


def index_sample(x, index):
    """根据index选择input的值

    Arags：
        input: 输入
        index: 索引矩阵

    Returns:
        经mask选择过的input

    >>> input
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> index
    [
        [1, 2],
        [0, 1]
    ]
    >>> index_sample(input, index)
    [
        [2, 3],
        [4, 5]
    ]
    """
    x_s = x.shape
    dim = len(index.shape) - 1
    assert x_s[:dim] == index.shape[:dim]
    r_x = layers.reshape(x, shape=(-1, *x_s[dim:]))
    index = layers.reshape(index, shape=(len(r_x), -1, 1))
    # generate arange index, shape like index
    arr_index = layers.arange(start=0, end=len(index), dtype=index.dtype)
    arr_index = layers.unsqueeze(arr_index, axes=[1, 2])
    arr_index = layers.expand_as(arr_index, index)
    #  genrate new index
    new_index = layers.concat((arr_index, index), -1)
    new_index = layers.reshape(new_index, (-1, 2))
    # get output
    out = layers.gather_nd(r_x, new_index)
    out = layers.reshape(out, (*x_s[:dim], -1))
    return out


def mask_fill(input, mask, value):
    """根据mask中为True的索引将value填充到mask中对应的位置

    Args:
        input: 输入矩阵
        mask: mask矩阵
        value: 填充值

    Returns:
        返回填充value后的input

    >>> input
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
    >>> mask
    [
        [True, True, False],
        [True, False, False]
    ]
    >>> mask_fill(input, mask, 0)
    [
        [1, 2, 0],
        [4, 0, 0]
    ]
    """
    return input * layers.cast(layers.logical_not(
        mask), input.dtype) + layers.cast(mask, input.dtype) * value


def unsqueeze(input, axes):
    """增加input的维度"""
    input_dtype = input.dtype
    if input_dtype == VarDesc.VarType.BOOL:
        input = layers.cast(input, 'int32')
    output = layers.unsqueeze(input, axes=axes)
    if input_dtype == VarDesc.VarType.BOOL:
        output = layers.cast(output, 'bool')
    return output


def reduce_sum(input, dim):
    """沿着input的dim维做sum"""
    input_dtype = input.dtype
    if input_dtype == VarDesc.VarType.BOOL:
        input = layers.cast(input, 'int32')
    output = layers.reduce_sum(input, dim=dim)
    return output
