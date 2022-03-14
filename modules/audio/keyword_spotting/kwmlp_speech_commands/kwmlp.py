# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PostNorm(nn.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class SpatialGatingUnit(nn.Layer):
    def __init__(self, dim, dim_seq, act=nn.Identity(), init_eps=1e-3):
        super().__init__()
        dim_out = dim // 2

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1D(dim_seq, dim_seq, 1)

        self.act = act

        init_eps /= dim_seq

    def forward(self, x):
        res, gate = x.split(2, axis=-1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        gate = F.conv1d(gate, weight, bias)

        return self.act(gate) * res


class gMLPBlock(nn.Layer):
    def __init__(self, *, dim, dim_ff, seq_len, act=nn.Identity()):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(dim, dim_ff), nn.GELU())

        self.sgu = SpatialGatingUnit(dim_ff, seq_len, act)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)
        return x


class Rearrange(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.transpose([0, 1, 3, 2]).squeeze(1)
        return x


class Reduce(nn.Layer):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        x = x.mean(axis=self.axis, keepdim=False)
        return x


class KW_MLP(nn.Layer):
    """Keyword-MLP."""

    def __init__(self,
                 input_res=[40, 98],
                 patch_res=[40, 1],
                 num_classes=35,
                 dim=64,
                 depth=12,
                 ff_mult=4,
                 channels=1,
                 prob_survival=0.9,
                 pre_norm=False,
                 **kwargs):
        super().__init__()
        image_height, image_width = input_res
        patch_height, patch_width = patch_res
        assert (image_height % patch_height) == 0 and (
            image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        P_Norm = PreNorm if pre_norm else PostNorm

        dim_ff = dim * ff_mult

        self.to_patch_embed = nn.Sequential(Rearrange(), nn.Linear(channels * patch_height * patch_width, dim))

        self.prob_survival = prob_survival

        self.layers = nn.LayerList(
            [Residual(P_Norm(dim, gMLPBlock(dim=dim, dim_ff=dim_ff, seq_len=num_patches))) for i in range(depth)])

        self.to_logits = nn.Sequential(nn.LayerNorm(dim), Reduce(axis=1), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers
        x = nn.Sequential(*layers)(x)
        return self.to_logits(x)
