"""
Various utilities for neural networks implemented by Paddle. This code is rewritten based on:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
"""
import math

import paddle
import paddle.nn as nn


class SiLU(nn.Layer):

    def forward(self, x):
        return x * nn.functional.sigmoid(x)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1D(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2D(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1D(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2D(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = paddle.exp(-math.log(max_period) * paddle.arange(start=0, end=half, dtype=paddle.float32) / half)
    args = paddle.cast(timesteps[:, None], 'float32') * freqs[None]
    embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat([embedding, paddle.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    This function is disabled. And now just forward.
    """
    return func(*inputs)
