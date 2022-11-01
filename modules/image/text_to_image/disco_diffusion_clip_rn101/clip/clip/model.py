from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import nn

from .layers import AttentionPool2d
from .layers import Bottleneck
from .layers import MultiHeadAttention
from .layers import ResidualAttentionBlock


class ModifiedResNet(nn.Layer):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2D(3, width // 2, kernel_size=3, stride=2, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(width // 2)
        self.conv2 = nn.Conv2D(width // 2, width // 2, kernel_size=3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(width // 2)
        self.conv3 = nn.Conv2D(width // 2, width, kernel_size=3, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(width)
        self.avgpool = nn.AvgPool2D(2)
        self.relu = nn.ReLU()

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        #x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class Transformer(nn.Layer):

    def __init__(self, width: int, layers: int, heads: int, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisualTransformer(nn.Layer):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # used patch_size x patch_size, stride patch_size to do linear projection
        self.conv1 = nn.Conv2D(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias_attr=False)

        # scale = width ** -0.5
        self.class_embedding = paddle.create_parameter((width, ), 'float32')

        self.positional_embedding = paddle.create_parameter(((input_resolution // patch_size)**2 + 1, width), 'float32')

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = paddle.create_parameter((width, output_dim), 'float32')

    def forward(self, x):

        x = self.conv1(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.transpose((0, 2, 1))
        x = paddle.concat([self.class_embedding + paddle.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x], axis=1)

        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = paddle.matmul(x, self.proj)

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
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int):
        super().__init__()

        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(layers=vision_layers,
                                         output_dim=embed_dim,
                                         heads=vision_heads,
                                         input_resolution=image_resolution,
                                         width=vision_width)
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(input_resolution=image_resolution,
                                            patch_size=vision_patch_size,
                                            width=vision_width,
                                            layers=vision_layers,
                                            heads=vision_heads,
                                            output_dim=embed_dim)

        self.transformer = Transformer(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = paddle.create_parameter((self.context_length, transformer_width), 'float32')
        self.ln_final = nn.LayerNorm(transformer_width)

        self.text_projection = paddle.create_parameter((transformer_width, embed_dim), 'float32')
        self.logit_scale = paddle.create_parameter((1, ), 'float32')

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # mask = paddle.empty((self.context_length, self.context_length),dtype='float32')
        # mask.fill_(float("-inf"))
        #mask.triu_(1)  # zero out the lower diagonal

        mask = paddle.ones((self.context_length, self.context_length)) * float("-inf")
        mask = paddle.triu(mask, diagonal=1)

        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print(x.shape)

        x = x + self.positional_embedding
        #print(x.shape)

        x = x.transpose((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        idx = text.numpy().argmax(-1)
        idx = list(idx)
        x = [x[i:i + 1, int(j), :] for i, j in enumerate(idx)]
        x = paddle.concat(x, 0)
        x = paddle.matmul(x, self.text_projection)
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = paddle.matmul(logit_scale * image_features, text_features.t())
        logits_per_text = paddle.matmul(logit_scale * text_features, image_features.t())

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
