import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleclas.ppcls.arch.backbone.model_zoo.vision_transformer import VisionTransformer


class Slice(nn.Layer):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class AddReadout(nn.Layer):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)


class Transpose(nn.Layer):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        prems = list(range(x.dim()))
        prems[self.dim0], prems[self.dim1] = prems[self.dim1], prems[self.dim0]
        x = x.transpose(prems)
        return x


class Unflatten(nn.Layer):
    def __init__(self, start_axis, shape):
        super(Unflatten, self).__init__()
        self.start_axis = start_axis
        self.shape = shape

    def forward(self, x):
        return paddle.reshape(x, x.shape[:self.start_axis] + [self.shape])


class ProjectReadout(nn.Layer):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(
            nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = paddle.concat((x[:, self.start_index:], readout), -1)

        return self.project(features)


class ViT(VisionTransformer):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, class_num=1000,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                 qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
                 norm_layer='nn.LayerNorm', epsilon=1e-6, **kwargs):
        super().__init__(img_size, patch_size, in_chans, class_num, embed_dim,
                         depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate,
                         attn_drop_rate, drop_path_rate, norm_layer, epsilon, **kwargs)
        self.patch_size = patch_size
        self.start_index = 1
        features = [256, 512, 1024, 1024]
        readout_oper = [
            ProjectReadout(embed_dim, self.start_index) for out_feat in features
        ]
        self.act_postprocess1 = nn.Sequential(
            readout_oper[0],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2D(
                in_channels=embed_dim,
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2DTranspose(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess2 = nn.Sequential(
            readout_oper[1],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2D(
                in_channels=embed_dim,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2DTranspose(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                dilation=1,
                groups=1,
            ),
        )

        self.act_postprocess3 = nn.Sequential(
            readout_oper[2],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2D(
                in_channels=embed_dim,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        self.act_postprocess4 = nn.Sequential(
            readout_oper[3],
            Transpose(1, 2),
            Unflatten(2, [img_size // 16, img_size // 16]),
            nn.Conv2D(
                in_channels=embed_dim,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2D(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.norm = nn.Identity()
        self.head = nn.Identity()

    def _resize_pos_embed(self, posemb, gs_h, gs_w):
        posemb_tok, posemb_grid = (
            posemb[:, : self.start_index],
            posemb[0, self.start_index:],
        )

        gs_old = int(math.sqrt(len(posemb_grid)))

        posemb_grid = posemb_grid.reshape(
            (1, gs_old, gs_old, -1)).transpose((0, 3, 1, 2))
        posemb_grid = F.interpolate(
            posemb_grid, size=(gs_h, gs_w), mode="bilinear")
        posemb_grid = posemb_grid.transpose(
            (0, 2, 3, 1)).reshape((1, gs_h * gs_w, -1))

        posemb = paddle.concat([posemb_tok, posemb_grid], axis=1)

        return posemb

    def forward(self, x):
        b, c, h, w = x.shape

        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size, w // self.patch_size
        )
        x = self.patch_embed.proj(x).flatten(2).transpose((0, 2, 1))

        cls_tokens = self.cls_token.expand(
            (b, -1, -1)
        )
        x = paddle.concat((cls_tokens, x), axis=1)

        x = x + pos_embed
        x = self.pos_drop(x)

        outputs = []
        for index, blk in enumerate(self.blocks):
            x = blk(x)
            if index in [5, 11, 17, 23]:
                outputs.append(x)

        layer_1 = self.act_postprocess1[0:2](outputs[0])
        layer_2 = self.act_postprocess2[0:2](outputs[1])
        layer_3 = self.act_postprocess3[0:2](outputs[2])
        layer_4 = self.act_postprocess4[0:2](outputs[3])

        shape = (-1, 1024, h // self.patch_size, w // self.patch_size)
        layer_1 = layer_1.reshape(shape)
        layer_2 = layer_2.reshape(shape)
        layer_3 = layer_3.reshape(shape)
        layer_4 = layer_4.reshape(shape)

        layer_1 = self.act_postprocess1[3: len(self.act_postprocess1)](layer_1)
        layer_2 = self.act_postprocess2[3: len(self.act_postprocess2)](layer_2)
        layer_3 = self.act_postprocess3[3: len(self.act_postprocess3)](layer_3)
        layer_4 = self.act_postprocess4[3: len(self.act_postprocess4)](layer_4)

        return layer_1, layer_2, layer_3, layer_4
