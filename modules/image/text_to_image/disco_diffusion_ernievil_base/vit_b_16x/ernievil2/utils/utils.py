import json
import os
from typing import List
from typing import Union

import numpy as np
import paddle
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.clip_vision_transformer import ViT_base_patch16_224
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.clip_vision_transformer import ViT_base_patch32_224
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.clip_vision_transformer import ViT_large_patch14_224
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.efficientnet import EfficientNetB5
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.ernie2 import ErnieModel
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.transformers.multimodal import MultiModalModel
from disco_diffusion_ernievil_base.vit_b_16x.ernievil2.utils.tokenizer import FullTokenizer

__all__ = ['tokenize', 'build_model']

MODEL_NAMES = ['vit_b_16x']

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
_tokenizer = FullTokenizer(vocab_file=os.path.join(os.path.dirname(__file__),
                                                   '../../packages/ernie_base_3.0/vocab.txt'),
                           do_lower_case=True)


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


def build_model(name='vit_b_16x'):
    assert name in MODEL_NAMES, f"model name must be one of {MODEL_NAMES}"
    name2model = {'vit_b_16x': build_vit_b_16x_model}
    model = name2model[name]()
    return model


def build_vit_b_16x_model():
    # Define model
    image_model = ViT_base_patch16_224()
    with open(os.path.join(os.path.dirname(__file__),
                           '../../packages/ernie_base_3.0/ernie_config.base.json')) as json_file:
        config_dict = json.load(json_file)
    text_model = ErnieModel(config_dict)
    model = MultiModalModel(image_model, text_model)
    checkpoint = paddle.load(os.path.join(os.path.dirname(__file__), '../../pre_trained/vit_b_16x.pdparams'))
    model.set_state_dict(checkpoint)
    model.eval()
    return model
