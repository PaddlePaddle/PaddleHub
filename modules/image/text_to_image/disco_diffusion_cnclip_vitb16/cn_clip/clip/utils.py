# Code modified from https://github.com/openai/CLIP
import json
import os
from pathlib import Path
from typing import List
from typing import Union

import paddle
from disco_diffusion_cnclip_vitb16.cn_clip.clip import _tokenizer
from disco_diffusion_cnclip_vitb16.cn_clip.clip.model import CLIP
from tqdm import tqdm

__all__ = ["tokenize", "create_model", "available_models"]

_MODEL_INFO = {"ViTB16": {"struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese", "input_resolution": 224}}


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODEL_INFO.keys())


def tokenize(texts: Union[str, List[str]], context_length: int = 64):
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 24 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] +
                          _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[:context_length - 2] +
                          [_tokenizer.vocab['[SEP]']])

    result = paddle.zeros([len(all_tokens), context_length], dtype='int64')

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = paddle.to_tensor(tokens)

    return result


def create_model(name):
    checkpoint = paddle.load(os.path.join(os.path.dirname(__file__), 'pre_trained', '{}.pdparams'.format(name)))
    model_name = _MODEL_INFO[name]['struct']
    vision_model, text_model = model_name.split('@')
    # Initialize the model.
    vision_model_config_file = Path(__file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(__file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info)
    model.set_state_dict(checkpoint)
    return model
