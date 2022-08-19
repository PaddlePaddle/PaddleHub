from collections import OrderedDict
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
from disco_diffusion_cnclip_vitb16.cn_clip.clip import _tokenizer
from disco_diffusion_cnclip_vitb16.cn_clip.clip.configuration_bert import BertConfig
from disco_diffusion_cnclip_vitb16.cn_clip.clip.modeling_bert import BertModel
from paddle import nn
from paddle.nn import MultiHeadAttention


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2D(inplanes, planes, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)

        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.avgpool = nn.AvgPool2D(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2D(planes, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([("-1", nn.AvgPool2D(stride)),
                             ("0", nn.Conv2D(inplanes, planes * self.expansion, 1, stride=1, bias_attr=False)),
                             ("1", nn.BatchNorm2D(planes * self.expansion))]))

    def forward(self, x: paddle.Tensor):
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


class QuickGELU(nn.Layer):

    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Layer):

    def __init__(self, d_model: int, n_head: int, attn_mask: paddle.Tensor = None):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(*[("c_fc", nn.Linear(d_model, d_model * 4)), (
            "gelu", QuickGELU()), ("c_proj", nn.Linear(d_model * 4, d_model))])
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: paddle.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, attn_mask=self.attn_mask)

    def forward(self, x: paddle.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Layer):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: paddle.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: paddle.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Layer):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2D(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias_attr=False)

        scale = width**-0.5
        # self.class_embedding = nn.Parameter(scale * paddle.randn(width))
        class_embedding = self.create_parameter([width])
        self.add_parameter("class_embedding", class_embedding)
        # self.positional_embedding = nn.Parameter(scale * paddle.randn([(input_resolution // patch_size) ** 2 + 1, width)])
        positional_embedding = self.create_parameter([(input_resolution // patch_size)**2 + 1, width])
        self.add_parameter("positional_embedding", positional_embedding)
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        # self.proj = nn.Parameter(scale * paddle.randn([width, output_dim]))
        proj = self.create_parameter([width, output_dim])
        self.add_parameter("proj", proj)

    def forward(self, x: paddle.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape([x.shape[0], x.shape[1], -1])  # shape = [*, width, grid ** 2]
        x = x.transpose([0, 2, 1])  # shape = [*, grid ** 2, width]
        x = paddle.concat([self.class_embedding + paddle.zeros([x.shape[0], 1, x.shape[-1]], dtype=x.dtype), x],
                          axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + paddle.cast(self.positional_embedding, x.dtype)
        x = self.ln_pre(x)

        x = self.transformer(x)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Layer):

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        vocab_size: int,
        text_attention_probs_dropout_prob: float,
        text_hidden_act: str,
        text_hidden_dropout_prob: float,
        text_hidden_size: int,
        text_initializer_range: float,
        text_intermediate_size: int,
        text_max_position_embeddings: int,
        text_num_attention_heads: int,
        text_num_hidden_layers: int,
        text_type_vocab_size: int,
        tokenizer=_tokenizer,
    ):
        super().__init__()

        vision_heads = vision_width // 64
        self.visual = VisualTransformer(input_resolution=image_resolution,
                                        patch_size=vision_patch_size,
                                        width=vision_width,
                                        layers=vision_layers,
                                        heads=vision_heads,
                                        output_dim=embed_dim)

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
        )
        self.bert = BertModel(self.bert_config)

        text_projection = self.create_parameter([text_hidden_size, embed_dim])
        self.add_parameter("text_projection", text_projection)
        logit_scale = self.create_parameter([1])
        self.add_parameter("logit_scale", logit_scale)

        self.tokenizer = tokenizer

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.cast(self.dtype))

    def encode_text(self, text):
        pad_index = self.tokenizer.vocab['[PAD]']

        attn_mask = text.not_equal(paddle.to_tensor(pad_index)).cast(self.dtype)

        x = self.bert(text, attention_mask=attn_mask)[0].cast(self.dtype)  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection

    def forward(self, image, text):
        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / text_features.norm(axis=-1, keepdim=True)

        return image_features, text_features, self.logit_scale.exp()

    def get_similarity(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(axis=1, keepdim=True)
        text_features = text_features / text_features.norm(axis=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
