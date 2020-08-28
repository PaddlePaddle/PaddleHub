# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Dict, List, Optional, Union, Tuple
import json

import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid

from paddlehub.utils.log import logger
from paddlehub.module.module import moduleinfo
from paddlehub.tokenizer.bert_tokenizer import BertTokenizer


def _build_linear(n_in, n_out, name, init, act=None):
    return dygraph.Linear(
        n_in,
        n_out,
        param_attr=fluid.ParamAttr(name='%s.w_0' % name if name is not None else None, initializer=init),
        bias_attr='%s.b_0' % name if name is not None else None,
        act=act)


def _build_ln(n_in, name):
    return dygraph.LayerNorm(
        normalized_shape=n_in,
        param_attr=fluid.ParamAttr(
            name='%s_layer_norm_scale' % name if name is not None else None,
            initializer=fluid.initializer.Constant(1.)),
        bias_attr=fluid.ParamAttr(
            name='%s_layer_norm_bias' % name if name is not None else None, initializer=fluid.initializer.Constant(1.)))


def append_name(name, postfix):
    if name is None:
        return None
    elif name == '':
        return postfix
    else:
        return '%s_%s' % (name, postfix)


class AttentionLayer(dygraph.Layer):
    def __init__(self, cfg, name=None):
        super(AttentionLayer, self).__init__()
        initializer = fluid.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        n_head = cfg['num_attention_heads']
        assert d_model % n_head == 0
        d_model_q = cfg.get('query_hidden_size_per_head', d_model // n_head) * n_head
        d_model_v = cfg.get('value_hidden_size_per_head', d_model // n_head) * n_head
        self.n_head = n_head
        self.d_key = d_model_q // n_head
        self.q = _build_linear(d_model, d_model_q, append_name(name, 'query_fc'), initializer)
        self.k = _build_linear(d_model, d_model_q, append_name(name, 'key_fc'), initializer)
        self.v = _build_linear(d_model, d_model_v, append_name(name, 'value_fc'), initializer)
        self.o = _build_linear(d_model_v, d_model, append_name(name, 'output_fc'), initializer)
        self.dropout = lambda i: fluid.layers.dropout(
            i,
            dropout_prob=cfg['attention_probs_dropout_prob'],
            dropout_implementation="upscale_in_train",
        ) if self.training else i

    def forward(self, queries, keys, values, attn_bias, past_cache):
        assert len(queries.shape) == len(keys.shape) == len(values.shape) == 3
        #bsz, q_len, q_dim = queries.shape
        #bsz, k_len, k_dim = keys.shape
        #bsz, v_len, v_dim = values.shape
        #assert k_len == v_len

        q = self.q(queries)
        k = self.k(keys)
        v = self.v(values)

        cache = (k, v)
        if past_cache is not None:
            cached_k, cached_v = past_cache
            k = fluid.layers.concat([cached_k, k], 1)
            v = fluid.layers.concat([cached_v, v], 1)

        q = fluid.layers.transpose(
            fluid.layers.reshape(q, [0, 0, self.n_head, q.shape[-1] // self.n_head]),
            [0, 2, 1, 3])  #[batch, head, seq, dim]
        k = fluid.layers.transpose(
            fluid.layers.reshape(k, [0, 0, self.n_head, k.shape[-1] // self.n_head]),
            [0, 2, 1, 3])  #[batch, head, seq, dim]
        v = fluid.layers.transpose(
            fluid.layers.reshape(v, [0, 0, self.n_head, v.shape[-1] // self.n_head]),
            [0, 2, 1, 3])  #[batch, head, seq, dim]

        q = fluid.layers.scale(q, scale=self.d_key**-0.5)
        score = fluid.layers.matmul(q, k, transpose_y=True)
        if attn_bias is not None:
            score += attn_bias
        score = fluid.layers.softmax(score, use_cudnn=True)
        score = self.dropout(score)

        out = fluid.layers.matmul(score, v)
        out = fluid.layers.transpose(out, [0, 2, 1, 3])
        out = fluid.layers.reshape(out, [0, 0, out.shape[2] * out.shape[3]])

        out = self.o(out)
        return out, cache


class PositionwiseFeedForwardLayer(dygraph.Layer):
    def __init__(self, cfg, name=None):
        super(PositionwiseFeedForwardLayer, self).__init__()
        initializer = fluid.initializer.TruncatedNormal(scale=cfg['initializer_range'])
        d_model = cfg['hidden_size']
        d_ffn = cfg.get('intermediate_size', 4 * d_model)
        assert cfg['hidden_act'] in ['relu', 'gelu']
        self.i = _build_linear(d_model, d_ffn, append_name(name, 'fc_0'), initializer, act=cfg['hidden_act'])
        self.o = _build_linear(d_ffn, d_model, append_name(name, 'fc_1'), initializer)
        prob = cfg.get('intermediate_dropout_prob', 0.)
        self.dropout = lambda i: fluid.layers.dropout(
            i,
            dropout_prob=prob,
            dropout_implementation="upscale_in_train",
        ) if self.training else i

    def forward(self, inputs):
        hidden = self.i(inputs)
        hidden = self.dropout(hidden)
        out = self.o(hidden)
        return out


class ErnieBlock(dygraph.Layer):
    def __init__(self, cfg, name=None):
        super(ErnieBlock, self).__init__()
        d_model = cfg['hidden_size']
        initializer = fluid.initializer.TruncatedNormal(scale=cfg['initializer_range'])

        self.attn = AttentionLayer(cfg, name=append_name(name, 'multi_head_att'))
        self.ln1 = _build_ln(d_model, name=append_name(name, 'post_att'))
        self.ffn = PositionwiseFeedForwardLayer(cfg, name=append_name(name, 'ffn'))
        self.ln2 = _build_ln(d_model, name=append_name(name, 'post_ffn'))
        prob = cfg.get('intermediate_dropout_prob', cfg['hidden_dropout_prob'])
        self.dropout = lambda i: fluid.layers.dropout(
            i,
            dropout_prob=prob,
            dropout_implementation="upscale_in_train",
        ) if self.training else i

    def forward(self, inputs, attn_bias=None, past_cache=None):
        attn_out, cache = self.attn(inputs, inputs, inputs, attn_bias, past_cache=past_cache)  #self attn
        attn_out = self.dropout(attn_out)
        hidden = attn_out + inputs
        hidden = self.ln1(hidden)  # dropout/ add/ norm

        ffn_out = self.ffn(hidden)
        ffn_out = self.dropout(ffn_out)
        hidden = ffn_out + hidden
        hidden = self.ln2(hidden)
        return hidden, cache


class ErnieEncoderStack(dygraph.Layer):
    """
    The model of ERNIE.
    """

    def __init__(self, cfg, name=None):
        super(ErnieEncoderStack, self).__init__()
        n_layers = cfg['num_hidden_layers']
        self.block = dygraph.LayerList([ErnieBlock(cfg, append_name(name, 'layer_%d' % i)) for i in range(n_layers)])

    def forward(self, inputs, attn_bias=None, past_cache=None):
        if past_cache is not None:
            assert isinstance(
                past_cache,
                tuple), 'unknown type of `past_cache`, expect tuple or list. got %s' % repr(type(past_cache))
            past_cache = list(zip(*past_cache))
        else:
            past_cache = [None] * len(self.block)
        cache_list_k, cache_list_v, hidden_list = [], [], [inputs]

        for b, p in zip(self.block, past_cache):
            inputs, cache = b(inputs, attn_bias=attn_bias, past_cache=p)
            cache_k, cache_v = cache
            cache_list_k.append(cache_k)
            cache_list_v.append(cache_v)
            hidden_list.append(inputs)

        return inputs, hidden_list, (cache_list_k, cache_list_v)


@moduleinfo(
    name="ernie",
    version="2.0.0",
    summary=
    "Baidu's ERNIE, Enhanced Representation through kNowledge IntEgration, max_seq_len=512 when predtrained. The module is executed as paddle.dygraph.",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    type="nlp/semantic_model")
class ErnieModel(dygraph.Layer):
    """
    Fundamental pretrained Ernie model
    """

    def __init__(self):
        name = None
        config_path = "/mnt/zhangxuefei/program-paddle/PaddleHub/hub_module/modules/text/semantic_model/ernie_dygraph/model_params/ernie_config.json"
        with open(config_path, 'r', encoding='utf8') as json_file:
            cfg = json.load(json_file)
        dygraph.Layer.__init__(self)
        d_model = cfg['hidden_size']
        d_emb = cfg.get('emb_size', cfg['hidden_size'])
        d_vocab = cfg['vocab_size']
        d_pos = cfg['max_position_embeddings']
        d_sent = cfg.get("sent_type_vocab_size") or cfg['type_vocab_size']
        self.n_head = cfg['num_attention_heads']
        initializer = fluid.initializer.TruncatedNormal(scale=cfg['initializer_range'])

        self.ln = _build_ln(d_model, name=append_name(name, 'pre_encoder'))
        self.word_emb = dygraph.Embedding([d_vocab, d_emb],
                                          param_attr=fluid.ParamAttr(
                                              name=append_name(name, 'word_embedding'), initializer=initializer))
        self.pos_emb = dygraph.Embedding([d_pos, d_emb],
                                         param_attr=fluid.ParamAttr(
                                             name=append_name(name, 'pos_embedding'), initializer=initializer))
        self.sent_emb = dygraph.Embedding([d_sent, d_emb],
                                          param_attr=fluid.ParamAttr(
                                              name=append_name(name, 'sent_embedding'), initializer=initializer))
        prob = cfg['hidden_dropout_prob']
        self.dropout = lambda i: fluid.layers.dropout(
            i,
            dropout_prob=prob,
            dropout_implementation="upscale_in_train",
        ) if self.training else i

        self.encoder_stack = ErnieEncoderStack(cfg, append_name(name, 'encoder'))
        if cfg.get('has_pooler', True):
            self.pooler = _build_linear(
                cfg['hidden_size'], cfg['hidden_size'], append_name(name, 'pooled_fc'), initializer, act='tanh')
        else:
            self.pooler = None

        state_dict, _ = fluid.load_dygraph(self.get_pretrained_params())
        for k, v in self.state_dict().items():
            if k not in state_dict:
                logger.warning('param:%s not set in pretrained model, skip' % k)
                state_dict[k] = v

        self.set_dict(state_dict)
        logger.info("%s pretrained parameters have been loaded!" % self.name)

    def forward(self,
                src_ids: fluid.Variable,
                sent_ids: fluid.Variable,
                pos_ids: fluid.Variable,
                input_mask: fluid.Variable,
                attn_bias: Union[fluid.Variable, bool] = None,
                past_cache: List[List[fluid.Variable]] = None,
                use_causal_mask: bool = False):
        """
        Args:
            src_ids (:obj:`Variable` of shape `[batch_size, seq_len]`):
                Indices of input sequence tokens in the vocabulary.
            sent_ids (:obj: `Variable` of shape `[batch_size, seq_len]`):
                Aka token_type_ids, Segment token indices to indicate first and second portions of the inputs.
                if None, assume all tokens come from `segment_a`
            pos_ids(:obj: `Variable` of shape `[batch_size, seq_len]`):
                Indices of positions of each input sequence tokens in the position embeddings.
            input_mask(:obj: `Variable` of shape `[batch_size, seq_len]`):
                Mask to avoid performing attention on the padding token indices of the encoder input.
            attn_bias(:obj: `Variable` of shape `[batch_size, seq_len, seq_len]` or :obj: False, optional, defaults to :obj: `None`):
                3D version of `input_mask`, if set, overrides `input_mask`; if set not False, will not apply attention mask
            past_cache(optional, tuple of two lists: cached key and cached value,
                each is a list of `Variable`s of shape `[batch_size, seq_len, hidden_size]`):
                cached key/value tensor that will be concated to generated key/value when performing self attention.
                if set, `attn_bias` should not be None.

        Returns:
            pooled (:obj: `Variable` of shape `[batch_size, hidden_size]`):
                Output logits of pooler classifier
            encoded(`Variable` of shape `[batch_size, seq_len, hidden_size]`):
                Output logits of transformer stack
        """
        #d_batch, d_seqlen = src_ids.shape
        assert len(src_ids.shape) == 2, 'expect src_ids.shape = [batch, sequence], got %s' % (repr(src_ids.shape))
        d_batch = fluid.layers.shape(src_ids)[0]
        d_seqlen = fluid.layers.shape(src_ids)[1]
        if pos_ids is None:
            pos_ids = fluid.layers.reshape(fluid.layers.range(0, d_seqlen, 1, dtype='int32'), [1, -1])
            pos_ids = fluid.layers.cast(pos_ids, 'int64')

        assert len(input_mask.shape) == 2
        input_mask = fluid.layers.unsqueeze(input_mask, axes=[-1])
        input_mask = fluid.layers.cast(input_mask, 'float32')
        attn_bias = fluid.layers.matmul(input_mask, input_mask, transpose_y=True)

        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = fluid.layers.unsqueeze(attn_bias, [1])
        attn_bias = fluid.layers.expand(attn_bias, [1, self.n_head, 1, 1])  # avoid broadcast =_=
        attn_bias.stop_gradient = True

        if sent_ids is None:
            sent_ids = fluid.layers.zeros_like(src_ids)

        src_embedded = self.word_emb(src_ids)
        pos_embedded = self.pos_emb(pos_ids)
        sent_embedded = self.sent_emb(sent_ids)
        embedded = src_embedded + pos_embedded + sent_embedded

        embedded = self.dropout(self.ln(embedded))

        encoded, hidden_list, cache_list = self.encoder_stack(embedded, attn_bias, past_cache=None)
        if self.pooler is not None:
            pooled = self.pooler(encoded[:, 0, :])
        else:
            pooled = None

        return pooled, encoded

    def get_vocab_path(self):
        """
        Gets the path of the module vocabulary path.
        """
        vocab_path = "/mnt/zhangxuefei/program-paddle/PaddleHub_wzw/hub_module/modules/text/semantic_model/ernie_dygraph/model_params/vocab.txt"
        return vocab_path

    def get_pretrained_params(self):
        """
        Gets the path of the module pretrained parameters.
        """
        param_path = '/mnt/zhangxuefei/program-paddle/PaddleHub_wzw/hub_module/modules/text/semantic_model/ernie_dygraph/model_params/saved_weights.pdparams'
        return param_path

    def get_tokenizer(self, tokenize_chinese_chars=True):
        """
        Gets the tokenizer that is customized for this module.

        Args:
            tokenize_chinese_chars (:obj: bool , defaults to :obj: True):
                Whether to tokenize chinese characters or not.
        Returns:
            tokenizer (:obj:BertTokenizer) : The tokenizer which was customized for this module.
        """
        return BertTokenizer(tokenize_chinese_chars=tokenize_chinese_chars, vocab_file=self.get_vocab_path())


if __name__ == "__main__":
    import numpy as np
    src_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    sent_ids = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
    pos_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    input_mask = np.array([[0, 0, 0, 0, 0]], dtype=np.int64)
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        ernie = ErnieModel()
        src_ids = dygraph.to_variable(src_ids)
        sent_ids = dygraph.to_variable(sent_ids)
        pos_ids = dygraph.to_variable(pos_ids)
        input_mask = dygraph.to_variable(input_mask)
        pooled, encoded = ernie(src_ids, sent_ids, pos_ids, input_mask)
        print(pooled.shape)
