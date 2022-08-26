# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import math
from inspect import isfunction

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


paddle.finfo = finfo


class AttentionBlockNew(nn.Layer):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention
    """

    def __init__(
        self,
        channels,
        num_head_channels=None,
        num_groups=32,
        rescale_output_factor=1.0,
        eps=1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, epsilon=eps)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels)

    def transpose_for_scores(self, projection: paddle.Tensor) -> paddle.Tensor:
        new_projection_shape = projection.shape[:-1] + [self.num_heads, -1]
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.reshape(new_projection_shape).transpose([0, 2, 1, 3])
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.reshape([batch, channel, height * width]).transpose([0, 2, 1])

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = paddle.matmul(query_states * scale, key_states * scale, transpose_y=True)
        attention_probs = F.softmax(attention_scores.astype("float32"), axis=-1).astype(attention_scores.dtype)

        # compute attention output
        context_states = paddle.matmul(attention_probs, value_states)

        context_states = context_states.transpose([0, 2, 1, 3])
        new_context_states_shape = context_states.shape[:-2] + [
            self.channels,
        ]
        context_states = context_states.reshape(new_context_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(context_states)
        hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch, channel, height, width])

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    def set_weight(self, attn_layer):
        self.group_norm.weight.set_value(attn_layer.norm.weight)
        self.group_norm.bias.set_value(attn_layer.norm.bias)

        if hasattr(attn_layer, "q"):
            self.query.weight.set_value(attn_layer.q.weight[:, :, 0, 0])
            self.key.weight.set_value(attn_layer.k.weight[:, :, 0, 0])
            self.value.weight.set_value(attn_layer.v.weight[:, :, 0, 0])

            self.query.bias.set_value(attn_layer.q.bias)
            self.key.bias.set_value(attn_layer.k.bias)
            self.value.bias.set_value(attn_layer.v.bias)

            self.proj_attn.weight.set_value(attn_layer.proj_out.weight[:, :, 0, 0])
            self.proj_attn.bias.set_value(attn_layer.proj_out.bias)
        elif hasattr(attn_layer, "NIN_0"):
            self.query.weight.set_value(attn_layer.NIN_0.W.t())
            self.key.weight.set_value(attn_layer.NIN_1.W.t())
            self.value.weight.set_value(attn_layer.NIN_2.W.t())

            self.query.bias.set_value(attn_layer.NIN_0.b)
            self.key.bias.set_value(attn_layer.NIN_1.b)
            self.value.bias.set_value(attn_layer.NIN_2.b)

            self.proj_attn.weight.set_value(attn_layer.NIN_3.W.t())
            self.proj_attn.bias.set_value(attn_layer.NIN_3.b)

            self.group_norm.weight.set_value(attn_layer.GroupNorm_0.weight)
            self.group_norm.bias.set_value(attn_layer.GroupNorm_0.bias)
        else:
            qkv_weight = attn_layer.qkv.weight.reshape(
                [self.num_heads, 3 * self.channels // self.num_heads, self.channels])
            qkv_bias = attn_layer.qkv.bias.reshape([self.num_heads, 3 * self.channels // self.num_heads])

            q_w, k_w, v_w = qkv_weight.split(self.channels // self.num_heads, axis=1)
            q_b, k_b, v_b = qkv_bias.split(self.channels // self.num_heads, axis=1)

            self.query.weight.set_value(q_w.reshape([-1, self.channels]))
            self.key.weight.set_value(k_w.reshape([-1, self.channels]))
            self.value.weight.set_value(v_w.reshape([-1, self.channels]))

            self.query.bias.set_value(q_b.flatten())
            self.key.bias.set_value(k_b.flatten())
            self.value.bias.set_value(v_b.flatten())

            self.proj_attn.weight.set_value(attn_layer.proj.weight[:, :, 0])
            self.proj_attn.bias.set_value(attn_layer.proj.bias)


class SpatialTransformer(nn.Layer):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-6)

        self.proj_in = nn.Conv2D(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.LayerList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for d in range(depth)
        ])

        self.proj_out = nn.Conv2D(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.transpose([0, 2, 3, 1]).reshape([b, h * w, c])
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = x.reshape([b, h, w, c]).transpose([0, 3, 1, 2])
        x = self.proj_out(x)
        return x + x_in

    def set_weight(self, layer):
        self.norm = layer.norm
        self.proj_in = layer.proj_in
        self.transformer_blocks = layer.transformer_blocks
        self.proj_out = layer.proj_out


class BasicTransformerBlock(nn.Layer):

    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim,
                                    context_dim=context_dim,
                                    heads=n_heads,
                                    dim_head=d_head,
                                    dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class CrossAttention(nn.Layer):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias_attr=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape([batch_size, seq_len, head_size, dim // head_size])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape([batch_size * head_size, seq_len, dim // head_size])
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape([batch_size // head_size, head_size, seq_len, dim])
        tensor = tensor.transpose([0, 2, 1, 3]).reshape([batch_size // head_size, seq_len, dim * head_size])
        return tensor

    def forward(self, x, context=None, mask=None):
        batch_size, sequence_length, dim = x.shape

        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        sim = paddle.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = mask.reshape([batch_size, -1])
            max_neg_value = -paddle.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = F.softmax(sim, axis=-1)

        out = paddle.einsum("b i j, b j d -> b i d", attn, v)
        out = self.reshape_batch_dim_to_heads(out)
        return self.to_out(out)


class FeedForward(nn.Layer):

    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


# feedforward
class GEGLU(nn.Layer):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * F.gelu(gate)


# TODO(Patrick) - remove once all weights have been converted -> not needed anymore then
class NIN(nn.Layer):

    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = self.create_parameter(shape=[in_dim, num_units], default_initializer=nn.initializer.Constant(0.))
        self.b = self.create_parameter(shape=[
            num_units,
        ],
                                       is_bias=True,
                                       default_initializer=nn.initializer.Constant(0.))


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# the main attention block that is used for all models
class AttentionBlock(nn.Layer):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=None,
        num_groups=32,
        encoder_channels=None,
        overwrite_qkv=False,
        overwrite_linear=False,
        rescale_output_factor=1.0,
        eps=1e-5,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (channels % num_head_channels == 0
                    ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, epsilon=eps)
        self.qkv = nn.Conv1D(channels, channels * 3, 1)
        self.n_heads = self.num_heads
        self.rescale_output_factor = rescale_output_factor

        if encoder_channels is not None:
            self.encoder_kv = nn.Conv1D(encoder_channels, channels * 2, 1)

        self.proj = nn.Conv1D(channels, channels, 1)

        self.overwrite_qkv = overwrite_qkv
        self.overwrite_linear = overwrite_linear

        if overwrite_qkv:
            in_channels = channels
            self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, epsilon=1e-6)
            self.q = nn.Conv2D(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.k = nn.Conv2D(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.v = nn.Conv2D(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.proj_out = nn.Conv2D(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.overwrite_linear:
            num_groups = min(channels // 4, 32)
            self.norm = nn.GroupNorm(num_channels=channels, num_groups=num_groups, epsilon=1e-6)
            self.NIN_0 = NIN(channels, channels)
            self.NIN_1 = NIN(channels, channels)
            self.NIN_2 = NIN(channels, channels)
            self.NIN_3 = NIN(channels, channels)

            self.GroupNorm_0 = nn.GroupNorm(num_groups=num_groups, num_channels=channels, epsilon=1e-6)
        else:
            self.proj_out = nn.Conv1D(channels, channels, 1)
            self.set_weights(self)

        self.is_overwritten = False

    def set_weights(self, layer):
        if self.overwrite_qkv:
            qkv_weight = paddle.concat([layer.q.weight, layer.k.weight, layer.v.weight], axis=0)[:, :, :, 0]
            qkv_bias = paddle.concat([layer.q.bias, layer.k.bias, layer.v.bias], axis=0)

            self.qkv.weight.set_value(qkv_weight)
            self.qkv.bias.set_value(qkv_bias)

            proj_out = nn.Conv1D(self.channels, self.channels, 1)
            proj_out.weight.set_value(layer.proj_out.weight[:, :, :, 0])
            proj_out.bias.set_value(layer.proj_out.bias)

            self.proj = proj_out
        elif self.overwrite_linear:
            self.qkv.weight.set_value(
                paddle.concat([self.NIN_0.W.t(), self.NIN_1.W.t(), self.NIN_2.W.t()], axis=0)[:, :, None])
            self.qkv.bias.set_value(paddle.concat([self.NIN_0.b, self.NIN_1.b, self.NIN_2.b], axis=0))

            self.proj.weight.set_value(self.NIN_3.W.t()[:, :, None])
            self.proj.bias.set_value(self.NIN_3.b)

            self.norm.weight.set_value(self.GroupNorm_0.weight)
            self.norm.bias.set_value(self.GroupNorm_0.bias)
        else:
            self.proj.weight.set_value(self.proj_out.weight)
            self.proj.bias.set_value(self.proj_out.bias)

    def forward(self, x, encoder_out=None):
        if not self.is_overwritten and (self.overwrite_qkv or self.overwrite_linear):
            self.set_weights(self)
            self.is_overwritten = True

        b, c, *spatial = x.shape
        hid_states = self.norm(x).reshape([b, c, -1])

        qkv = self.qkv(hid_states)
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape([bs * self.n_heads, ch * 3, length]).split(ch, axis=1)

        if encoder_out is not None:
            encoder_kv = self.encoder_kv(encoder_out)
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape([bs * self.n_heads, ch * 2, -1]).split(ch, axis=1)
            k = paddle.concat([ek, k], axis=-1)
            v = paddle.concat([ev, v], axis=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum("bct,bcs->bts", q * scale, k * scale)  # More stable with f16 than dividing afterwards
        weight = F.softmax(weight.astype("float32"), axis=-1).astype(weight.dtype)

        a = paddle.einsum("bts,bcs->bct", weight, v)
        h = a.reshape([bs, -1, length])

        h = self.proj(h)
        h = h.reshape([b, c, *spatial])

        result = x + h

        result = result / self.rescale_output_factor

        return result
