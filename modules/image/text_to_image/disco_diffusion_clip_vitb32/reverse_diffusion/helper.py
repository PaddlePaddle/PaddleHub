'''
This code is rewritten by Paddle based on Jina-ai/discoart.
https://github.com/jina-ai/discoart/blob/main/discoart/helper.py
'''
import hashlib
import logging
import os
import subprocess
import sys
from os.path import expanduser
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import paddle


def _get_logger():
    logger = logging.getLogger(__package__)
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    ch.setLevel("INFO")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = _get_logger()


def load_clip_models(enabled: List[str], clip_models: Dict[str, Any] = {}):

    import disco_diffusion_clip_vitb32.clip.clip as clip
    from disco_diffusion_clip_vitb32.clip.clip import build_model, tokenize, transform

    # load enabled models
    for k in enabled:
        if k not in clip_models:
            clip_models[k] = build_model(name=k)
            clip_models[k].eval()
            for parameter in clip_models[k].parameters():
                parameter.stop_gradient = True

    # disable not enabled models to save memory
    for k in clip_models:
        if k not in enabled:
            clip_models.pop(k)

    return list(clip_models.values())


def load_all_models(diffusion_model, use_secondary_model):
    from .model.script_util import (
        model_and_diffusion_defaults, )

    model_config = model_and_diffusion_defaults()

    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': False,
            'use_scale_shift_norm': True,
        })
    elif diffusion_model == '256x256_diffusion_uncond':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': False,
            'use_scale_shift_norm': True,
        })

    secondary_model = None
    if use_secondary_model:
        from .model.sec_diff import SecondaryDiffusionImageNet2
        secondary_model = SecondaryDiffusionImageNet2()
        model_dict = paddle.load(
            os.path.join(os.path.dirname(__file__), 'pre_trained', 'secondary_model_imagenet_2.pdparams'))
        secondary_model.set_state_dict(model_dict)
        secondary_model.eval()
        for parameter in secondary_model.parameters():
            parameter.stop_gradient = True

    return model_config, secondary_model


def load_diffusion_model(model_config, diffusion_model, steps):
    from .model.script_util import (
        create_model_and_diffusion, )

    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    model_config.update({
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    model, diffusion = create_model_and_diffusion(**model_config)
    model.set_state_dict(
        paddle.load(os.path.join(os.path.dirname(__file__), 'pre_trained', f'{diffusion_model}.pdparams')))
    model.eval()
    for name, param in model.named_parameters():
        param.stop_gradient = True

    return model, diffusion


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])
