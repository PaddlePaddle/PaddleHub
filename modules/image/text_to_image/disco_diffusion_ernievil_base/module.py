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
import sys
from functools import partial
from typing import List
from typing import Optional

import paddle
from disco_diffusion_ernievil_base import resize_right
from disco_diffusion_ernievil_base.reverse_diffusion import create
from disco_diffusion_ernievil_base.vit_b_16x import ernievil2

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


@moduleinfo(name="disco_diffusion_ernievil_base",
            version="1.0.0",
            type="image/text_to_image",
            summary="",
            author="paddlepaddle",
            author_email="paddle-dev@baidu.com")
class DiscoDiffusionClip:

    def generate_image(self,
                       text_prompts,
                       style: Optional[str] = None,
                       artist: Optional[str] = None,
                       init_image: Optional[str] = None,
                       width_height: Optional[List[int]] = [1280, 768],
                       skip_steps: Optional[int] = 0,
                       steps: Optional[int] = 250,
                       cut_ic_pow: Optional[int] = 1,
                       init_scale: Optional[int] = 1000,
                       clip_guidance_scale: Optional[int] = 5000,
                       tv_scale: Optional[int] = 0,
                       range_scale: Optional[int] = 0,
                       sat_scale: Optional[int] = 0,
                       cutn_batches: Optional[int] = 4,
                       diffusion_sampling_mode: Optional[str] = 'ddim',
                       perlin_init: Optional[bool] = False,
                       perlin_mode: Optional[str] = 'mixed',
                       seed: Optional[int] = None,
                       eta: Optional[float] = 0.8,
                       clamp_grad: Optional[bool] = True,
                       clamp_max: Optional[float] = 0.05,
                       randomize_class: Optional[bool] = True,
                       clip_denoised: Optional[bool] = False,
                       fuzzy_prompt: Optional[bool] = False,
                       rand_mag: Optional[float] = 0.05,
                       cut_overview: Optional[str] = '[12]*400+[4]*600',
                       cut_innercut: Optional[str] = '[4]*400+[12]*600',
                       cut_icgray_p: Optional[str] = '[0.2]*400+[0]*600',
                       display_rate: Optional[int] = 10,
                       n_batches: Optional[int] = 1,
                       batch_size: Optional[int] = 1,
                       batch_name: Optional[str] = '',
                       use_gpu: Optional[bool] = True,
                       output_dir: Optional[str] = 'disco_diffusion_ernievil_base_out'):
        """
        Create Disco Diffusion artworks and save the result into a DocumentArray.

        :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.

        :param init_image: Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion.
        :param style: Image style, such as oil paintings, if specified, style will be used to construct prompts.
        :param artist: Artist style, if specified, style will be used to construct prompts.
        :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
        :param skip_steps: Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
        :param steps: When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.
        :param cut_ic_pow: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
        :param init_scale: This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.
        :param clip_guidance_scale: CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well.
        :param tv_scale: Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising
        :param range_scale: Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.
        :param sat_scale: Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation.
        :param cutn_batches: Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
        :param diffusion_sampling_mode: Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.
        :param perlin_init: Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively.
        :param perlin_mode: sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects.
        :param seed: Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical.
        :param eta: eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects.
        :param clamp_grad: As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced.
        :param clamp_max: Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy.
        :param fuzzy_prompt: Controls whether to add multiple noisy prompts to the prompt losses. If True, can increase variability of image output. Experiment with this.
        :param rand_mag: Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.
        :param cut_overview: The schedule of overview cuts
        :param cut_innercut: The schedule of inner cuts
        :param cut_icgray_p: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
        :param display_rate: During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it’s better to set display_rate equal to steps, because displaying interim images does slow Colab down slightly.
        :param n_batches: This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings.
        :param batch_name: The name of the batch, the batch id will be named as "discoart-[batch_name]-seed". To avoid your artworks be overridden by other users, please use a unique name.
        :param use_gpu: whether to use gpu or not.
        :return: a DocumentArray object that has `n_batches` Documents
        """
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
                text_prompts += "，{}".format(style)
            if artist is not None:
                text_prompts += "，由{}所作".format(artist)
        elif isinstance(text_prompts, list):
            text_prompts[0] = text_prompts[0].rstrip(',.，。')
            if style is not None:
                text_prompts[0] += "，{}".format(style)
            if artist is not None:
                text_prompts[0] += "，由{}所作".format(artist)

        return create(text_prompts=text_prompts,
                      init_image=init_image,
                      width_height=width_height,
                      skip_steps=skip_steps,
                      steps=steps,
                      cut_ic_pow=cut_ic_pow,
                      init_scale=init_scale,
                      clip_guidance_scale=clip_guidance_scale,
                      tv_scale=tv_scale,
                      range_scale=range_scale,
                      sat_scale=sat_scale,
                      cutn_batches=cutn_batches,
                      diffusion_sampling_mode=diffusion_sampling_mode,
                      perlin_init=perlin_init,
                      perlin_mode=perlin_mode,
                      seed=seed,
                      eta=eta,
                      clamp_grad=clamp_grad,
                      clamp_max=clamp_max,
                      randomize_class=randomize_class,
                      clip_denoised=clip_denoised,
                      fuzzy_prompt=fuzzy_prompt,
                      rand_mag=rand_mag,
                      cut_overview=cut_overview,
                      cut_innercut=cut_innercut,
                      cut_icgray_p=cut_icgray_p,
                      display_rate=display_rate,
                      n_batches=n_batches,
                      batch_size=batch_size,
                      batch_name=batch_name,
                      clip_models=['vit_b_16x'],
                      output_dir=output_dir)

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
                                      init_image=args.init_image,
                                      width_height=args.width_height,
                                      skip_steps=args.skip_steps,
                                      steps=args.steps,
                                      cut_ic_pow=args.cut_ic_pow,
                                      init_scale=args.init_scale,
                                      clip_guidance_scale=args.clip_guidance_scale,
                                      tv_scale=args.tv_scale,
                                      range_scale=args.range_scale,
                                      sat_scale=args.sat_scale,
                                      cutn_batches=args.cutn_batches,
                                      diffusion_sampling_mode=args.diffusion_sampling_mode,
                                      perlin_init=args.perlin_init,
                                      perlin_mode=args.perlin_mode,
                                      seed=args.seed,
                                      eta=args.eta,
                                      clamp_grad=args.clamp_grad,
                                      clamp_max=args.clamp_max,
                                      randomize_class=args.randomize_class,
                                      clip_denoised=args.clip_denoised,
                                      fuzzy_prompt=args.fuzzy_prompt,
                                      rand_mag=args.rand_mag,
                                      cut_overview=args.cut_overview,
                                      cut_innercut=args.cut_innercut,
                                      cut_icgray_p=args.cut_icgray_p,
                                      display_rate=args.display_rate,
                                      n_batches=args.n_batches,
                                      batch_size=args.batch_size,
                                      batch_name=args.batch_name,
                                      output_dir=args.output_dir)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_input_group.add_argument(
            '--skip_steps',
            type=int,
            default=0,
            help=
            'Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15%% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50%% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture'
        )
        self.arg_input_group.add_argument(
            '--steps',
            type=int,
            default=250,
            help=
            "When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time."
        )
        self.arg_input_group.add_argument(
            '--cut_ic_pow',
            type=int,
            default=1,
            help=
            "This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details."
        )
        self.arg_input_group.add_argument(
            '--init_scale',
            type=int,
            default=1000,
            help=
            "This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost."
        )
        self.arg_input_group.add_argument(
            '--clip_guidance_scale',
            type=int,
            default=5000,
            help=
            "CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well."
        )
        self.arg_input_group.add_argument(
            '--tv_scale',
            type=int,
            default=0,
            help=
            "Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising"
        )
        self.arg_input_group.add_argument(
            '--range_scale',
            type=int,
            default=0,
            help=
            "Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images."
        )
        self.arg_input_group.add_argument(
            '--sat_scale',
            type=int,
            default=0,
            help=
            "Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation."
        )
        self.arg_input_group.add_argument(
            '--cutn_batches',
            type=int,
            default=4,
            help=
            "Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below."
        )
        self.arg_input_group.add_argument(
            '--diffusion_sampling_mode',
            type=str,
            default='ddim',
            help=
            "Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord."
        )
        self.arg_input_group.add_argument(
            '--perlin_init',
            type=bool,
            default=False,
            help=
            "Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively."
        )
        self.arg_input_group.add_argument(
            '--perlin_mode',
            type=str,
            default='mixed',
            help=
            "sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects."
        )
        self.arg_input_group.add_argument(
            '--seed',
            type=int,
            default=None,
            help=
            "Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical."
        )
        self.arg_input_group.add_argument(
            '--eta',
            type=float,
            default=0.8,
            help=
            "eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects."
        )
        self.arg_input_group.add_argument(
            '--clamp_grad',
            type=bool,
            default=True,
            help=
            "As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced."
        )
        self.arg_input_group.add_argument(
            '--clamp_max',
            type=float,
            default=0.05,
            help=
            "Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy."
        )
        self.arg_input_group.add_argument('--randomize_class', type=bool, default=True, help="Random class.")
        self.arg_input_group.add_argument('--clip_denoised', type=bool, default=False, help="Clip denoised.")
        self.arg_input_group.add_argument(
            '--fuzzy_prompt',
            type=bool,
            default=False,
            help=
            "Controls whether to add multiple noisy prompts to the prompt losses. If True, can increase variability of image output. Experiment with this."
        )
        self.arg_input_group.add_argument(
            '--rand_mag',
            type=float,
            default=0.5,
            help="Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.")
        self.arg_input_group.add_argument('--cut_overview',
                                          type=str,
                                          default='[12]*400+[4]*600',
                                          help="The schedule of overview cuts")
        self.arg_input_group.add_argument('--cut_innercut',
                                          type=str,
                                          default='[4]*400+[12]*600',
                                          help="The schedule of inner cuts")
        self.arg_input_group.add_argument(
            '--cut_icgray_p',
            type=str,
            default='[0.2]*400+[0]*600',
            help=
            "This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details."
        )
        self.arg_input_group.add_argument(
            '--display_rate',
            type=int,
            default=10,
            help=
            "During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it’s better to set display_rate equal to steps, because displaying interim images does slow Colab down slightly."
        )
        self.arg_config_group.add_argument('--use_gpu',
                                           type=ast.literal_eval,
                                           default=True,
                                           help="whether use GPU or not")
        self.arg_config_group.add_argument('--output_dir',
                                           type=str,
                                           default='disco_diffusion_ernievil_base_out',
                                           help='Output directory.')

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--text_prompts', type=str)
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
            '--init_image',
            type=str,
            default=None,
            help=
            "Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion."
        )
        self.arg_input_group.add_argument(
            '--width_height',
            type=ast.literal_eval,
            default=[1280, 768],
            help=
            "Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so."
        )
        self.arg_input_group.add_argument(
            '--n_batches',
            type=int,
            default=1,
            help=
            "This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings."
        )
        self.arg_input_group.add_argument('--batch_size', type=int, default=1, help="Batch size.")
        self.arg_input_group.add_argument(
            '--batch_name',
            type=str,
            default='',
            help=
            'The name of the batch, the batch id will be named as "discoart-[batch_name]-seed". To avoid your artworks be overridden by other users, please use a unique name.'
        )
