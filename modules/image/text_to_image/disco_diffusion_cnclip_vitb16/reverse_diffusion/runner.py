'''
This code is rewritten by Paddle based on Jina-ai/discoart.
https://github.com/jina-ai/discoart/blob/main/discoart/runner.py
'''
import gc
import os
import random
from threading import Thread

import disco_diffusion_cnclip_vitb16.cn_clip.clip as clip
import numpy as np
import paddle
import paddle.vision.transforms as T
import paddle_lpips as lpips
from docarray import Document
from docarray import DocumentArray
from IPython import display
from ipywidgets import Output
from PIL import Image

from .helper import logger
from .helper import parse_prompt
from .model.losses import range_loss
from .model.losses import spherical_dist_loss
from .model.losses import tv_loss
from .model.make_cutouts import MakeCutoutsDango
from .model.sec_diff import alpha_sigma_to_t
from .model.sec_diff import SecondaryDiffusionImageNet2
from .model.transforms import Normalize


def do_run(args, models) -> 'DocumentArray':
    logger.info('preparing models...')
    model, diffusion, clip_models, secondary_model = models
    normalize = Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    lpips_model = lpips.LPIPS(net='vgg')
    for parameter in lpips_model.parameters():
        parameter.stop_gradient = True
    side_x = (args.width_height[0] // 64) * 64
    side_y = (args.width_height[1] // 64) * 64
    cut_overview = eval(args.cut_overview)
    cut_innercut = eval(args.cut_innercut)
    cut_icgray_p = eval(args.cut_icgray_p)

    from .model.perlin_noises import create_perlin_noise, regen_perlin

    seed = args.seed

    skip_steps = args.skip_steps

    loss_values = []

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        paddle.seed(seed)

    model_stats = []
    for clip_model in clip_models:
        model_stat = {
            'clip_model': None,
            'target_embeds': [],
            'make_cutouts': None,
            'weights': [],
        }
        model_stat['clip_model'] = clip_model

        if isinstance(args.text_prompts, str):
            args.text_prompts = [args.text_prompts]

        for prompt in args.text_prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(clip.tokenize(prompt))
            if args.fuzzy_prompt:
                for i in range(25):
                    model_stat['target_embeds'].append((txt + paddle.randn(txt.shape) * args.rand_mag).clip(0, 1))
                    model_stat['weights'].append(weight)
            else:
                model_stat['target_embeds'].append(txt)
                model_stat['weights'].append(weight)

        model_stat['target_embeds'] = paddle.concat(model_stat['target_embeds'])
        model_stat['weights'] = paddle.to_tensor(model_stat['weights'])
        if model_stat['weights'].sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        model_stat['weights'] /= model_stat['weights'].sum().abs()
        model_stats.append(model_stat)

    init = None
    if args.init_image:
        d = Document(uri=args.init_image).load_uri_to_image_tensor(side_x, side_y)
        init = T.to_tensor(d.tensor).unsqueeze(0) * 2 - 1

    if args.perlin_init:
        if args.perlin_mode == 'color':
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_y, side_x)
        elif args.perlin_mode == 'gray':
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_y, side_x)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x)
        else:
            init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1, False, side_y, side_x)
            init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x)
        init = (T.to_tensor(init).add(T.to_tensor(init2)).divide(paddle.to_tensor(2.0)).unsqueeze(0) * 2 - 1)
        del init2

    cur_t = None

    def cond_fn(x, t, y=None):
        x_is_NaN = False
        n = x.shape[0]
        if secondary_model:
            alpha = paddle.to_tensor(diffusion.sqrt_alphas_cumprod[cur_t], dtype='float32')
            sigma = paddle.to_tensor(diffusion.sqrt_one_minus_alphas_cumprod[cur_t], dtype='float32')
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            x = paddle.to_tensor(x.detach(), dtype='float32')
            x.stop_gradient = False
            cosine_t = paddle.tile(paddle.to_tensor(cosine_t.detach().cpu().numpy()), [n])
            cosine_t.stop_gradient = False
            out = secondary_model(x, cosine_t).pred
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in_d = out * fac + x * (1 - fac)
            x_in = x_in_d.detach()
            x_in.stop_gradient = False
            x_in_grad = paddle.zeros_like(x_in, dtype='float32')
        else:
            t = paddle.ones([n], dtype='int64') * cur_t
            out = diffusion.p_mean_variance(model, x, t, clip_denoised=False, model_kwargs={'y': y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in_d = out['pred_xstart'] * fac + x * (1 - fac)
            x_in = x_in_d.detach()
            x_in.stop_gradient = False
            x_in_grad = paddle.zeros_like(x_in, dtype='float32')
        for model_stat in model_stats:
            for i in range(args.cutn_batches):
                t_int = (int(t.item()) + 1)  # errors on last step without +1, need to find source
                # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                try:
                    input_resolution = model_stat['clip_model'].visual.input_resolution
                except:
                    input_resolution = 224

                cuts = MakeCutoutsDango(
                    input_resolution,
                    Overview=cut_overview[1000 - t_int],
                    InnerCrop=cut_innercut[1000 - t_int],
                    IC_Size_Pow=args.cut_ic_pow,
                    IC_Grey_P=cut_icgray_p[1000 - t_int],
                )
                clip_in = normalize(cuts(x_in.add(paddle.to_tensor(1.0)).divide(paddle.to_tensor(2.0))))
                image_embeds = (model_stat['clip_model'].encode_image(clip_in))

                dists = spherical_dist_loss(
                    image_embeds.unsqueeze(1),
                    model_stat['target_embeds'].unsqueeze(0),
                )

                dists = dists.reshape([
                    cut_overview[1000 - t_int] + cut_innercut[1000 - t_int],
                    n,
                    -1,
                ])
                losses = dists.multiply(model_stat['weights']).sum(2).mean(0)
                loss_values.append(losses.sum().item())  # log loss, probably shouldn't do per cutn_batch

                x_in_grad += (paddle.grad(losses.sum() * args.clip_guidance_scale, x_in)[0] / args.cutn_batches)
        tv_losses = tv_loss(x_in)
        range_losses = range_loss(x_in)
        sat_losses = paddle.abs(x_in - x_in.clip(min=-1, max=1)).mean()
        loss = (tv_losses.sum() * args.tv_scale + range_losses.sum() * args.range_scale +
                sat_losses.sum() * args.sat_scale)
        if init is not None and args.init_scale:
            init_losses = lpips_model(x_in, init)
            loss = loss + init_losses.sum() * args.init_scale
        x_in_grad += paddle.grad(loss, x_in)[0]
        if not paddle.isnan(x_in_grad).any():
            grad = -paddle.grad(x_in_d, x, x_in_grad)[0]
        else:
            x_is_NaN = True
            grad = paddle.zeros_like(x)
        if args.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (grad * magnitude.clip(max=args.clamp_max) / magnitude)
        return grad

    if args.diffusion_sampling_mode == 'ddim':
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    logger.info('creating artwork...')

    image_display = Output()
    da_batches = DocumentArray()

    for _nb in range(args.n_batches):
        display.clear_output(wait=True)
        display.display(args.name_docarray, image_display)
        gc.collect()
        paddle.device.cuda.empty_cache()

        d = Document(tags=vars(args))
        da_batches.append(d)

        cur_t = diffusion.num_timesteps - skip_steps - 1

        if args.perlin_init:
            init = regen_perlin(args.perlin_mode, side_y, side_x, args.batch_size)

        if args.diffusion_sampling_mode == 'ddim':
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                eta=args.eta,
            )
        else:
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                order=2,
            )

        threads = []
        for j, sample in enumerate(samples):
            cur_t -= 1
            with image_display:
                if j % args.display_rate == 0 or cur_t == -1:
                    for _, image in enumerate(sample['pred_xstart']):
                        image = (image + 1) / 2
                        image = image.clip(0, 1).squeeze().transpose([1, 2, 0]).numpy() * 255
                        image = np.uint8(image)
                        image = Image.fromarray(image)

                        image.save(os.path.join(args.output_dir, 'progress-{}.png'.format(_nb)))
                        c = Document(tags={'cur_t': cur_t})
                        c.load_pil_image_to_datauri(image)
                        d.chunks.append(c)
                        display.clear_output(wait=True)
                        display.display(display.Image(os.path.join(args.output_dir, 'progress-{}.png'.format(_nb))))
                        d.chunks.plot_image_sprites(os.path.join(args.output_dir,
                                                                 f'{args.name_docarray}-progress-{_nb}.png'),
                                                    show_index=True)
                        t = Thread(
                            target=_silent_push,
                            args=(
                                da_batches,
                                args.name_docarray,
                            ),
                        )
                        threads.append(t)
                        t.start()

                    if cur_t == -1:
                        d.load_pil_image_to_datauri(image)

        for t in threads:
            t.join()
    display.clear_output(wait=True)
    logger.info(f'done! {args.name_docarray}')
    da_batches.plot_image_sprites(skip_empty=True, show_index=True, keep_aspect_ratio=True)
    return da_batches


def _silent_push(da_batches: DocumentArray, name: str) -> None:
    try:
        da_batches.push(name)
    except Exception as ex:
        logger.debug(f'push failed: {ex}')
