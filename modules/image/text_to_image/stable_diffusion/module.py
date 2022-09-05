# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import ast
import os
import random
import sys
from functools import partial
from typing import List
from typing import Optional

import numpy as np
import paddle
from docarray import Document
from docarray import DocumentArray
from IPython import display
from PIL import Image
from stable_diffusion.clip.clip.utils import build_model
from stable_diffusion.clip.clip.utils import tokenize
from stable_diffusion.diffusers import AutoencoderKL
from stable_diffusion.diffusers import DDIMScheduler
from stable_diffusion.diffusers import LMSDiscreteScheduler
from stable_diffusion.diffusers import PNDMScheduler
from stable_diffusion.diffusers import UNet2DConditionModel
from tqdm.auto import tqdm

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="stable_diffusion",
            version="1.0.0",
            type="image/text_to_image",
            summary="",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com")
class StableDiffusion:

    def __init__(self):
        self.vae = AutoencoderKL(in_channels=3,
                                 out_channels=3,
                                 down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D",
                                                   "DownEncoderBlock2D"),
                                 up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D",
                                                 "UpDecoderBlock2D"),
                                 block_out_channels=(128, 256, 512, 512),
                                 layers_per_block=2,
                                 act_fn="silu",
                                 latent_channels=4,
                                 sample_size=512)

        self.unet = UNet2DConditionModel(sample_size=64,
                                         in_channels=4,
                                         out_channels=4,
                                         center_input_sample=False,
                                         flip_sin_to_cos=True,
                                         freq_shift=0,
                                         down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                                           "CrossAttnDownBlock2D", "DownBlock2D"),
                                         up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                                                         "CrossAttnUpBlock2D"),
                                         block_out_channels=(320, 640, 1280, 1280),
                                         layers_per_block=2,
                                         downsample_padding=1,
                                         mid_block_scale_factor=1,
                                         act_fn="silu",
                                         norm_num_groups=32,
                                         norm_eps=1e-5,
                                         cross_attention_dim=768,
                                         attention_head_dim=8)

        unet_path = os.path.join(self.directory, 'pre_trained', 'stable-diffusion-v1-4-unet.pdparams')
        vae_path = os.path.join(self.directory, 'pre_trained', 'stable-diffusion-v1-4-vae.pdparams')
        self.unet.set_dict(paddle.load(unet_path))
        self.vae.set_dict(paddle.load(vae_path))
        for parameter in self.unet.parameters():
            parameter.stop_gradient = True
        self.unet.eval()
        for parameter in self.vae.parameters():
            parameter.stop_gradient = True
        self.vae.eval()

        self.text_encoder = build_model()
        for parameter in self.text_encoder.parameters():
            parameter.stop_gradient = True
        self.scheduler = PNDMScheduler(beta_start=0.00085,
                                       beta_end=0.012,
                                       beta_schedule="scaled_linear",
                                       num_train_timesteps=1000,
                                       skip_prk_steps=True)

    def generate_image(self,
                       text_prompts,
                       style: Optional[str] = None,
                       artist: Optional[str] = None,
                       width_height: Optional[List[int]] = [512, 512],
                       batch_size: Optional[int] = 1,
                       num_inference_steps=50,
                       guidance_scale=7.5,
                       enable_fp16=False,
                       seed=None,
                       display_rate=5,
                       use_gpu=True,
                       output_dir: Optional[str] = 'stable_diffusion_out'):
        """
        Create Disco Diffusion artworks and save the result into a DocumentArray.

        :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.
        :param style: Image style, such as oil paintings, if specified, style will be used to construct prompts.
        :param artist: Artist style, if specified, style will be used to construct prompts.
        :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
        :param batch_size: This variable sets the number of still images you want SD to create for each prompt.
        :param num_inference_steps: The number of inference steps.
        :param guidance_scale: Increase the adherence to the conditional signal which in this case is text as well as overall sample quality.
        :param enable_fp16: Whether to use float16.
        :param use_gpu: whether to use gpu or not.
        :param output_dir: Output directory.
        :return: a DocumentArray object that has `n_batches` Documents
        """
        if seed:
            np.random.seed(seed)
            random.seed(seed)
            paddle.seed(seed)

        if use_gpu:
            try:
                _places = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if _places:
                    paddle.device.set_device("gpu:{}".format(0))
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES as cuda_device_id."
                )
        else:
            paddle.device.set_device("cpu")
        paddle.disable_static()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if isinstance(text_prompts, str):
            text_prompts = text_prompts.rstrip(',.，。')
            if style is not None:
                text_prompts += ",{}".format(style)
            if artist is not None:
                text_prompts += ",{},trending on artstation".format(artist)
            text_prompts = [text_prompts]
        elif isinstance(text_prompts, list):
            for i, prompt in enumerate(
                    text_prompts):  # different from dd here, dd can have multiple prompts for one image with weight.
                text_prompts[i] = prompt.rstrip(',.，。')
                if style is not None:
                    text_prompts[i] += ",{}".format(style)
                if artist is not None:
                    text_prompts[i] += ",{},trending on artstation".format(artist)

        width, height = width_height
        da_batches = DocumentArray()

        for prompt in text_prompts:
            d = Document(tags={'prompt': prompt})
            da_batches.append(d)
            for i in range(batch_size):
                d.chunks.append(Document(tags={'prompt': prompt, 'image idx': i}))
            d.chunks.append(Document(tags={'prompt': prompt, 'image idx': 'merged'}))
            with paddle.amp.auto_cast(enable=enable_fp16, level='O2'):
                prompts = [prompt] * batch_size
                text_input = tokenize(prompts)
                text_embeddings = self.text_encoder(text_input)
                uncond_input = tokenize([""] * batch_size)
                uncond_embeddings = self.text_encoder(uncond_input)
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings])

                latents = paddle.randn((batch_size, self.unet.in_channels, height // 8, width // 8), )
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    latents = latents * self.scheduler.sigmas[0]

                self.scheduler.set_timesteps(num_inference_steps)
                for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = paddle.concat([latents] * 2)

                    if isinstance(self.scheduler, LMSDiscreteScheduler):
                        sigma = self.scheduler.sigmas[i]
                        latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if isinstance(self.scheduler, LMSDiscreteScheduler):
                        latents = self.scheduler.step(noise_pred, i, latents)["prev_sample"]
                    else:
                        latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
                    if i % display_rate == 0:
                        # vae decode
                        images = self.vae.decode(1 / 0.18215 * latents)
                        images = (images / 2 + 0.5).clip(0, 1)
                        merge_image = images.cpu().transpose([2, 0, 3, 1]).flatten(1, 2).numpy()
                        merge_image = (merge_image * 255).round().astype(np.uint8)
                        merge_image = Image.fromarray(merge_image)
                        merge_image.save(os.path.join(output_dir, f'{prompt}-progress.png'))
                        c = Document(tags={'step': i, 'prompt': prompt})
                        c.load_pil_image_to_datauri(merge_image)
                        d.chunks[-1].chunks.append(c)
                        display.clear_output(wait=True)
                        display.display(merge_image)
                        images = images.cpu().transpose([0, 2, 3, 1]).numpy()
                        images = (images * 255).round().astype(np.uint8)
                        for j in range(images.shape[0]):
                            image = Image.fromarray(images[j])
                            c = Document(tags={'step': i, 'prompt': prompt})
                            c.load_pil_image_to_datauri(image)
                            d.chunks[j].chunks.append(c)

                # vae decode
                images = self.vae.decode(1 / 0.18215 * latents)
                images = (images / 2 + 0.5).clip(0, 1)
                merge_image = images.cpu().transpose([2, 0, 3, 1]).flatten(1, 2).numpy()
                merge_image = (merge_image * 255).round().astype(np.uint8)
                merge_image = Image.fromarray(merge_image)
                merge_image.save(os.path.join(output_dir, f'{prompt}-merge.png'))
                display.clear_output(wait=True)
                display.display(merge_image)
                d.load_pil_image_to_datauri(merge_image)
                d.chunks[-1].load_pil_image_to_datauri(merge_image)
                images = images.cpu().transpose([0, 2, 3, 1]).numpy()
                images = (images * 255).round().astype(np.uint8)
                for j in range(images.shape[0]):
                    image = Image.fromarray(images[j])
                    image.save(os.path.join(output_dir, f'{prompt}-image-{j}.png'))
                    d.chunks[j].load_pil_image_to_datauri(image)
        return da_batches

    @serving
    def serving_method(self, text_prompts, **kwargs):
        """
        Run as a service.
        """
        results = self.generate_image(text_prompts=text_prompts, **kwargs).to_base64()
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.generate_image(text_prompts=args.text_prompts,
                                      style=args.style,
                                      artist=args.artist,
                                      width_height=args.width_height,
                                      batch_size=args.batch_size,
                                      num_inference_steps=args.num_inference_steps,
                                      guidance_scale=args.guidance_scale,
                                      enable_fp16=args.enable_fp16,
                                      seed=args.seed,
                                      display_rate=args.display_rate,
                                      use_gpu=args.use_gpu,
                                      output_dir=args.output_dir)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_input_group.add_argument('--num_inference_steps',
                                          type=int,
                                          default=50,
                                          help="The number of inference steps.")

        self.arg_input_group.add_argument(
            '--guidance_scale',
            type=float,
            default=7.5,
            help=
            "Increase the adherence to the conditional signal which in this case is text as well as overall sample quality."
        )

        self.arg_input_group.add_argument(
            '--seed',
            type=int,
            default=None,
            help=
            "Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed."
        )

        self.arg_input_group.add_argument(
            '--display_rate',
            type=int,
            default=10,
            help="During a diffusion run, you can monitor the progress of each image being created with this variable.")

        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=True,
                                           help="whether use GPU or not")

        self.arg_config_group.add_argument('--enable_fp16',
                                           type=ast.literal_eval,
                                           default=False,
                                           help="whether use float16 or not")

        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='stable_diffusion_out',
                                           help='Output directory.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--text_prompts',
            type=str,
            help=
            'Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.'
        )
        self.arg_input_group.add_argument(
            '--style',
            type=str,
            default=None,
            help='Image style, such as oil paintings, if specified, style will be used to construct prompts.')
        self.arg_input_group.add_argument('--artist',
                                          type=str,
                                          default=None,
                                          help='Artist style, if specified, style will be used to construct prompts.')

        self.arg_input_group.add_argument(
            '--width_height',
            type=ast.literal_eval,
            default=[512, 512],
            help=
            "Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so."
        )
        self.arg_input_group.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help="This variable sets the number of still images you want SD to create for each prompt.")
