from typing import Optional

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import functional as F
from paddle.nn import Linear

__all__ = ['ResidualAttentionBlock', 'AttentionPool2d', 'multi_head_attention_forward', 'MultiHeadAttention']


def multi_head_attention_forward(x: Tensor,
                                 num_heads: int,
                                 q_proj: Linear,
                                 k_proj: Linear,
                                 v_proj: Linear,
                                 c_proj: Linear,
                                 attn_mask: Optional[Tensor] = None):
    max_len, batch_size, emb_dim = x.shape
    head_dim = emb_dim // num_heads
    scaling = float(head_dim)**-0.5
    q = q_proj(x)  # L, N, E
    k = k_proj(x)  # L, N, E
    v = v_proj(x)  # L, N, E
    #k = k.con
    v = v.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    k = k.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))
    q = q.reshape((-1, batch_size * num_heads, head_dim)).transpose((1, 0, 2))

    q = q * scaling
    qk = paddle.bmm(q, k.transpose((0, 2, 1)))
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask.unsqueeze_(0)
        #assert str(attn_mask.dtype) == 'VarType.FP32' and attn_mask.ndim == 3
        assert attn_mask.shape[0] == 1 and attn_mask.shape[1] == max_len and attn_mask.shape[2] == max_len
        qk += attn_mask

    qk = paddle.nn.functional.softmax(qk, axis=-1)
    atten = paddle.bmm(qk, v)
    atten = atten.transpose((1, 0, 2))
    atten = atten.reshape((max_len, batch_size, emb_dim))
    atten = c_proj(atten)
    return atten


class MultiHeadAttention(nn.Layer):  # without attention mask

    def __init__(self, emb_dim: int, num_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, emb_dim, bias_attr=True)
        self.k_proj = nn.Linear(emb_dim, emb_dim, bias_attr=True)
        self.v_proj = nn.Linear(emb_dim, emb_dim, bias_attr=True)
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias_attr=True)
        self.head_dim = emb_dim // num_heads
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        assert self.head_dim * num_heads == emb_dim, "embed_dim must be divisible by num_heads"
        #self.scaling = float(self.head_dim) ** -0.5

    def forward(self, x, attn_mask=None):  # x is in shape[max_len,batch_size,emb_dim]

        atten = multi_head_attention_forward(x,
                                             self.num_heads,
                                             self.q_proj,
                                             self.k_proj,
                                             self.v_proj,
                                             self.c_proj,
                                             attn_mask=attn_mask)

        return atten


class Identity(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU()
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(
                ("-1", nn.AvgPool2D(stride)),
                ("0", nn.Conv2D(inplanes, planes * self.expansion, 1, stride=1, bias_attr=False)),
                ("1", nn.BatchNorm2D(planes * self.expansion)))

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Layer):

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()

        self.positional_embedding = paddle.create_parameter((spacial_dim**2 + 1, embed_dim), dtype='float32')

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=True)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim, bias_attr=True)
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

    def forward(self, x):

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).transpose((2, 0, 1))  # NCHW -> (HW)NC
        max_len, batch_size, emb_dim = x.shape
        head_dim = self.head_dim
        x = paddle.concat([paddle.mean(x, axis=0, keepdim=True), x], axis=0)
        x = x + paddle.unsqueeze(self.positional_embedding, 1)
        out = multi_head_attention_forward(x, self.num_heads, self.q_proj, self.k_proj, self.v_proj, self.c_proj)

        return out[0]


class QuickGELU(nn.Layer):

    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Layer):

    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                                 ("c_proj", nn.Linear(d_model * 4, d_model)))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        x = self.attn(x, self.attn_mask)
        assert isinstance(x, paddle.Tensor)  # not tuble here
        return x

    def forward(self, x):

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
