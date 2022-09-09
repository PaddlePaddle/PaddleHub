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
import numpy as np
import paddle
import paddle.nn as nn

from ..configuration_utils import ConfigMixin
from ..configuration_utils import register_to_config
from .unet_blocks import get_down_block
from .unet_blocks import get_up_block
from .unet_blocks import UNetMidBlock2D


class Encoder(nn.Layer):

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", ),
            block_out_channels=(64, ),
            layers_per_block=2,
            act_fn="silu",
            double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2D(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.LayerList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=32,
            temb_channels=None,
        )

        # out
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=num_groups_out, epsilon=1e-6)
        self.conv_act = nn.Silu()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2D(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Layer):

    def __init__(
            self,
            in_channels=3,
            out_channels=3,
            up_block_types=("UpDecoderBlock2D", ),
            block_out_channels=(64, ),
            layers_per_block=2,
            act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2D(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.LayerList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=32,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = 32
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, epsilon=1e-6)
        self.conv_act = nn.Silu()
        self.conv_out = nn.Conv2D(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Layer):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", paddle.to_tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape([ishape[0], -1])
        used = self.used
        match = (inds[:, :, None] == used[None, None, ...]).astype("int64")
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = paddle.randint(0, self.re_embed, shape=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape([ishape[0], -1])
        used = self.used
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = paddle.gather(used[None, :][inds.shape[0] * [0], :], inds, axis=1)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.transpose([0, 2, 3, 1])
        z_flattened = z.reshape([-1, self.e_dim])
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (paddle.sum(z_flattened**2, axis=1, keepdim=True) + paddle.sum(self.embedding.weight**2, axis=1) -
             2 * paddle.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t()))

        min_encoding_indices = paddle.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * paddle.mean((z_q.detach() - z)**2) + paddle.mean((z_q - z.detach())**2)
        else:
            loss = paddle.mean((z_q.detach() - z)**2) + self.beta * paddle.mean((z_q - z.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.transpose([0, 3, 1, 2])

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape([z.shape[0], -1])  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape([-1, 1])  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape([z_q.shape[0], z_q.shape[2], z_q.shape[3]])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape([shape[0], -1])  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.flatten()  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.reshape(shape)
            # reshape back to match original input shape
            z_q = z_q.transpose([0, 3, 1, 2])

        return z_q


class DiagonalGaussianDistribution(object):

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * paddle.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        else:
            if other is None:
                return 0.5 * paddle.sum(paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar, axis=[1, 2, 3])
            else:
                return 0.5 * paddle.sum(
                    paddle.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar +
                    other.logvar,
                    axis=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(logtwopi + self.logvar + paddle.pow(sample - self.mean, 2) / self.var, axis=dims)

    def mode(self):
        return self.mean


class VQModel(ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", ),
        up_block_types=("UpDecoderBlock2D", ),
        block_out_channels=(64, ),
        layers_per_block=1,
        act_fn="silu",
        latent_channels=3,
        sample_size=32,
        num_vq_embeddings=256,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            double_z=False,
        )

        self.quant_conv = nn.Conv2D(latent_channels, latent_channels, 1)
        self.quantize = VectorQuantizer(num_vq_embeddings,
                                        latent_channels,
                                        beta=0.25,
                                        remap=None,
                                        sane_index_shape=False)
        self.post_quant_conv = nn.Conv2D(latent_channels, latent_channels, 1)

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def forward(self, sample):
        x = sample
        h = self.encode(x)
        dec = self.decode(h)
        return dec


class AutoencoderKL(nn.Layer, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        sample_size=512,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2D(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2D(latent_channels, latent_channels, 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, sample, sample_posterior=False):
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec
