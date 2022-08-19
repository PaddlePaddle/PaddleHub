# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import print_function

import sys
import warnings
from functools import partial
from functools import reduce

import paddle
from paddle.fluid import core
from paddle.fluid.data_feeder import check_dtype
from paddle.fluid.data_feeder import check_type
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.data_feeder import convert_dtype
from paddle.fluid.framework import default_main_program
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.layers import control_flow
from paddle.fluid.layers import nn
from paddle.fluid.layers import sequence_lod
from paddle.fluid.layers import tensor
from paddle.fluid.layers import utils
from paddle.fluid.layers.utils import *
from paddle.fluid.param_attr import ParamAttr
from paddle.utils import deprecated
#import paddle.nn as nn


class ArrayWrapper(object):

    def __init__(self, x):
        self.array = [x]

    def append(self, x):
        self.array.append(x)
        return self

    def __getitem__(self, item):
        return self.array.__getitem__(item)


def _maybe_copy(state, new_state, step_mask):
    """update rnn state or just pass the old state through"""
    new_state = nn.elementwise_mul(new_state, step_mask, axis=0) \
              + nn.elementwise_mul(state, (1 - step_mask), axis=0)
    return new_state


def _transpose_batch_time(x):
    perm = [1, 0] + list(range(2, len(x.shape)))
    return nn.transpose(x, perm)


class Decoder(object):
    """
	:api_attr: Static Graph

    Decoder is the base class for any decoder instance used in `dynamic_decode`.
    It provides interface for output generation for one time step, which can be
    used to generate sequences.

    The key abstraction provided by Decoder is:

    1. :code:`(initial_input, initial_state, finished) = initialize(inits)` ,
    which generates the input and state for the first decoding step, and gives the
    initial status telling whether each sequence in the batch is finished.
    It would be called once before the decoding iterations.

    2. :code:`(output, next_state, next_input, finished) = step(time, input, state)` ,
    which transforms the input and state to the output and new state, generates
    input for the next decoding step, and emits the flag indicating finished status.
    It is the main part for each decoding iteration.

    3. :code:`(final_outputs, final_state) = finalize(outputs, final_state, sequence_lengths)` ,
    which revises the outputs(stack of all time steps' output) and final state(state from the
    last decoding step) to get the counterpart for special usage.
    Not necessary to be implemented if no need to revise the stacked outputs and
    state from the last decoding step. If implemented, it would be called after
    the decoding iterations.

    Decoder is more general compared to RNNCell, since the returned `next_input`
    and `finished` make it can determine the input and when to finish by itself
    when used in dynamic decoding. Decoder always wraps a RNNCell instance though
    not necessary.
    """

    def initialize(self, inits):
        r"""
        Called once before the decoding iterations.

        Parameters:
            inits: Argument provided by the caller.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` and `initial_states` both are a (possibly nested \
                structure of) tensor variable[s], and `finished` is a tensor with \
                bool data type.
        """
        raise NotImplementedError

    def step(self, time, inputs, states, **kwargs):
        r"""
        Called per step of decoding.

        Parameters:
            time(Variable): A Tensor with shape :math:`[1]` provided by the caller.
                The data type is int64.
            inputs(Variable): A (possibly nested structure of) tensor variable[s].
            states(Variable): A (possibly nested structure of) tensor variable[s].
            **kwargs: Additional keyword arguments, provided by the caller.

        Returns:
            tuple: A tuple( :code:(outputs, next_states, next_inputs, finished)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s], and the structure, shape and \
                data type must be same as the counterpart from input arguments. \
                `outputs` is a (possibly nested structure of) tensor variable[s]. \
                `finished` is a Tensor with bool data type.
        """
        raise NotImplementedError

    def finalize(self, outputs, final_states, sequence_lengths):
        r"""
        Called once after the decoding iterations if implemented.

        Parameters:
            outputs(Variable): A (possibly nested structure of) tensor variable[s].
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape
                :math:`[time\_step, batch\_size, ...]` , which is done by the caller.
            final_states(Variable): A (possibly nested structure of) tensor variable[s].
                It is the `next_states` returned by `decoder.step` at last decoding step,
                thus has the same structure, shape and data type with states at any time
                step.

        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ). \
                `final_outputs` and `final_states` both are a (possibly nested \
                structure of) tensor variable[s].
        """
        raise NotImplementedError

    @property
    def tracks_own_finished(self):
        """
        Describes whether the Decoder keeps track of finished states by itself.

        `decoder.step()` would emit a bool `finished` value at each decoding
        step. The emited `finished` can be used to determine whether every
        batch entries is finished directly, or it can be combined with the
        finished tracker keeped in `dynamic_decode` by performing a logical OR
        to take the already finished into account.

        If `False`, the latter would be took when performing `dynamic_decode`,
        which is the default. Otherwise, the former would be took, which uses
        the finished value emited by the decoder as all batch entry finished
        status directly, and it is the case when batch entries might be
        reordered such as beams in BeamSearchDecoder.

        Returns:
            bool: A python bool `False`.
        """
        return False


class BeamSearchDecoder(Decoder):
    """
    Decoder with beam search decoding strategy. It wraps a cell to get probabilities,
    and follows a beam search step to calculate scores and select candidate
    token ids for each decoding step.

    Please refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    **NOTE** When decoding with beam search, the `inputs` and `states` of cell
    would be tiled to `beam_size` (unsqueeze and tile), resulting to shapes like
    `[batch_size * beam_size, ...]` , which is built into `BeamSearchDecoder` and
    done automatically. Thus any other tensor with shape `[batch_size, ...]` used
    in `cell.call` needs to be tiled manually first, which can be completed by using
    :code:`BeamSearchDecoder.tile_beam_merge_with_batch` . The most common case
    for this is the encoder output in attention mechanism.

    Returns:
        BeamSearchDecoder: An instance of decoder which can be used in \
            `paddle.nn.dynamic_decode` to implement decoding.

    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.nn import BeamSearchDecoder, dynamic_decode
            from paddle.nn import GRUCell, Linear, Embedding
            trg_embeder = Embedding(100, 32)
            output_layer = Linear(32, 32)
            decoder_cell = GRUCell(input_size=32, hidden_size=32)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)

    """

    def __init__(self, cell, start_token, end_token, beam_size, embedding_fn=None, output_fn=None):
        """
        Constructor of BeamSearchDecoder.

        Parameters:
            cell(RNNCellBase): An instance of `RNNCellBase` or object with the same interface.
            start_token(int): The start token id.
            end_token(int): The end token id.
            beam_size(int): The beam width used in beam search.
            embedding_fn(optional): A callable to apply to selected candidate ids.
                Mostly it is an embedding layer to transform ids to embeddings,
                and the returned value acts as the `input` argument for `cell.call`.
                If not provided, the id to embedding transformation must be built into
                `cell.call`. Default None.
            output_fn(optional): A callable to apply to the cell's output prior to
                calculate scores and select candidate token ids. Default None.
        """
        self.cell = cell
        self.embedding_fn = embedding_fn
        self.output_fn = output_fn
        self.start_token = start_token
        self.end_token = end_token
        self.beam_size = beam_size

    @staticmethod
    def tile_beam_merge_with_batch(x, beam_size):
        r"""
        Tile the batch dimension of a tensor. Specifically, this function takes
        a tensor t shaped `[batch_size, s0, s1, ...]` composed of minibatch
        entries `t[0], ..., t[batch_size - 1]` and tiles it to have a shape
        `[batch_size * beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            x(Variable): A tensor with shape `[batch_size, ...]`. The data type
                should be float32, float64, int32, int64 or bool.
            beam_size(int): The beam width used in beam search.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        check_type(x, 'x', (Variable), 'BeamSearchDecoder.tile_beam_merge_with_batch')
        x = nn.unsqueeze(x, [1])  # [batch_size, 1, ...]
        expand_times = [1] * len(x.shape)
        expand_times[1] = beam_size
        x = paddle.tile(x, expand_times)  # [batch_size, beam_size, ...]
        x = nn.transpose(x, list(range(2, len(x.shape))) + [0, 1])  # [..., batch_size, beam_size]
        # use 0 to copy to avoid wrong shape
        x = nn.reshape(x, shape=[0] * (len(x.shape) - 2) + [-1])  # [..., batch_size * beam_size]
        x = nn.transpose(x, [len(x.shape) - 1] + list(range(0, len(x.shape) - 1)))  # [batch_size * beam_size, ...]
        return x

    def _split_batch_beams(self, x):
        r"""
        Reshape a tensor with shape `[batch_size * beam_size, ...]` to a new
        tensor with shape `[batch_size, beam_size, ...]`.

        Parameters:
            x(Variable): A tensor with shape `[batch_size * beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.
        """
        check_type(x, 'x', (Variable), 'BeamSearchDecoder._split_batch_beams')
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=[-1, self.beam_size] + list(x.shape[1:]))

    def _merge_batch_beams(self, x):
        r"""
        Reshape a tensor with shape `[batch_size, beam_size, ...]` to a new
        tensor with shape `[batch_size * beam_size, ...]`.

        Parameters:
            x(Variable): A tensor with shape `[batch_size, beam_size, ...]`. The
                data type should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size * beam_size, ...]`, whose \
                data type is same as `x`.
        """
        check_type(x, 'x', (Variable), 'BeamSearchDecoder._merge_batch_beams')
        # TODO: avoid fake shape in compile-time like tile_beam_merge_with_batch
        return nn.reshape(x, shape=[-1] + list(x.shape[2:]))

    def _expand_to_beam_size(self, x):
        r"""
        This function takes a tensor t shaped `[batch_size, s0, s1, ...]` composed
        of minibatch entries `t[0], ..., t[batch_size - 1]` and tiles it to have a
        shape `[batch_size, beam_size, s0, s1, ...]` composed of minibatch entries
        `t[0], t[0], ..., t[1], t[1], ...` where each minibatch entry is repeated
        `beam_size` times.

        Parameters:
            x(Variable): A tensor with shape `[batch_size, ...]`, The data type
                should be float32, float64, int32, int64 or bool.

        Returns:
            Variable: A tensor with shape `[batch_size, beam_size, ...]`, whose \
                data type is same as `x`.
        """
        check_type(x, 'x', (Variable), 'BeamSearchDecoder._expand_to_beam_size')
        x = nn.unsqueeze(x, [1])
        expand_times = [1] * len(x.shape)
        expand_times[1] = self.beam_size
        x = paddle.tile(x, expand_times)
        return x

    def _mask_probs(self, probs, finished):
        r"""
        Mask log probabilities. It forces finished beams to allocate all probability
        mass to eos and unfinished beams to remain unchanged.

        Parameters:
            probs(Variable): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the log probabilities. Its data type should be float32 or float64.
            finished(Variable): A tensor with shape `[batch_size, beam_size]`,
                representing the finished status for all beams. Its data type
                should be bool.

        Returns:
            Variable: A tensor with the same shape and data type as `x`, \
                where unfinished beams stay unchanged and finished beams are \
                replaced with a tensor with all probability on the EOS token.
        """
        check_type(probs, 'probs', (Variable), 'BeamSearchDecoder._mask_probs')
        check_type(finished, 'finished', (Variable), 'BeamSearchDecoder._mask_probs')
        # TODO: use where_op
        finished = tensor.cast(finished, dtype=probs.dtype)
        probs = nn.elementwise_mul(paddle.tile(nn.unsqueeze(finished, [2]), [1, 1, self.vocab_size]),
                                   self.noend_mask_tensor,
                                   axis=-1) - nn.elementwise_mul(probs, (finished - 1), axis=0)
        return probs

    def _gather(self, x, indices, batch_size):
        r"""
        Gather from the tensor `x` using `indices`.

        Parameters:
            x(Variable): A tensor with shape `[batch_size, beam_size, ...]`.
            indices(Variable): A `int64` tensor with shape `[batch_size, beam_size]`,
                representing the indices that we use to gather.
            batch_size(Variable): A tensor with shape `[1]`. Its data type should
                be int32 or int64.

        Returns:
            Variable: A tensor with the same shape and data type as `x`, \
                representing the gathered tensor.
        """
        check_type(x, 'x', (Variable), 'BeamSearchDecoder._gather')
        check_type(indices, 'indices', (Variable), 'BeamSearchDecoder._gather')
        check_type(batch_size, 'batch_size', (Variable), 'BeamSearchDecoder._gather')
        # TODO: compatibility of int32 and int64
        batch_size = tensor.cast(batch_size, indices.dtype) if batch_size.dtype != indices.dtype else batch_size
        batch_size.stop_gradient = True  # TODO: remove this
        batch_pos = paddle.tile(nn.unsqueeze(tensor.range(0, batch_size, 1, dtype=indices.dtype), [1]),
                                [1, self.beam_size])
        topk_coordinates = nn.stack([batch_pos, indices], axis=2)
        topk_coordinates.stop_gradient = True
        return nn.gather_nd(x, topk_coordinates)

    class OutputWrapper(collections.namedtuple("OutputWrapper", ("scores", "predicted_ids", "parent_ids"))):
        """
        The structure for the returned value `outputs` of `decoder.step`.
        A namedtuple includes scores, predicted_ids, parent_ids as fields.
        """
        pass

    class StateWrapper(collections.namedtuple("StateWrapper", ("cell_states", "log_probs", "finished", "lengths"))):
        """
        The structure for the argument `states` of `decoder.step`.
        A namedtuple includes cell_states, log_probs, finished, lengths as fields.
        """
        pass

    def initialize(self, initial_cell_states, bos_ids=None):
        r"""
        Initialize the BeamSearchDecoder.

        Parameters:
            initial_cell_states(Variable): A (possibly nested structure of)
                tensor variable[s]. An argument provided by the caller.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` is a tensor t filled by `start_token` with shape \
                `[batch_size, beam_size]` when `embedding_fn` is None, or the \
                returned value of `embedding_fn(t)` when `embedding_fn` is provided. \
                `initial_states` is a nested structure(namedtuple including cell_states, \
                log_probs, finished, lengths as fields) of tensor variables, where \
                `log_probs, finished, lengths` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, bool, int64`. \
                cell_states has a value with the same structure as the input \
                argument `initial_cell_states` but with tiled shape `[batch_size, beam_size, ...]`. \
                `finished` is a `bool` tensor filled by False with shape `[batch_size, beam_size]`.
        """
        self.kinf = 1e9
        state = flatten(initial_cell_states)[0]
        self.batch_size = nn.shape(state)[0]

        if bos_ids is not None:
            self.start_token = bos_ids

        self.start_token_tensor = tensor.fill_constant(shape=[1], dtype="int64", value=self.start_token)
        self.end_token_tensor = tensor.fill_constant(shape=[1], dtype="int64", value=self.end_token)

        init_cell_states = map_structure(self._expand_to_beam_size, initial_cell_states)
        init_inputs = paddle.full(shape=[self.batch_size, self.beam_size],
                                  fill_value=self.start_token_tensor,
                                  dtype=self.start_token_tensor.dtype)
        log_probs = paddle.tile(tensor.assign(np.array([[0.] + [-self.kinf] * (self.beam_size - 1)], dtype="float32")),
                                [self.batch_size, 1])
        if paddle.get_default_dtype() == "float64":
            log_probs = tensor.cast(log_probs, "float64")
        # TODO: remove the restriction of force_cpu
        init_finished = tensor.fill_constant_batch_size_like(input=state,
                                                             shape=[-1, self.beam_size],
                                                             dtype="bool",
                                                             value=False,
                                                             force_cpu=True)
        init_lengths = tensor.zeros_like(init_inputs)
        init_inputs = self.embedding_fn(init_inputs) if self.embedding_fn else init_inputs
        return init_inputs, self.StateWrapper(init_cell_states, log_probs, init_finished, init_lengths), init_finished

    def _beam_search_step(self, time, logits, next_cell_states, beam_state):
        r"""
        Calculate scores and select candidate token ids.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            logits(Variable): A tensor with shape `[batch_size, beam_size, vocab_size]`,
                representing the logits at the current time step. Its data type is float32.
            next_cell_states(Variable): A (possibly nested structure of) tensor variable[s].
                It has the same structure, shape and data type as the `cell_states` of
                `initial_states` returned by `initialize()`. It represents the next state
                from the cell.
            beam_state(Variable): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.

        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state)` ). \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`.
                `beam_search_state` has the same structure, shape and data type \
                as the input argument `beam_state`.

        """
        self.vocab_size = logits.shape[-1]
        self.vocab_size_tensor = tensor.fill_constant(shape=[1], dtype="int64", value=self.vocab_size)
        noend_array = [-self.kinf] * self.vocab_size
        noend_array[self.end_token] = 0

        self.noend_mask_tensor = tensor.assign(np.array(noend_array, "float32"))
        if paddle.get_default_dtype() == "float64":
            self.noend_mask_tensor = tensor.cast(self.noend_mask_tensor, "float64")

        step_log_probs = nn.log(nn.softmax(logits))
        step_log_probs = self._mask_probs(step_log_probs, beam_state.finished)
        log_probs = nn.elementwise_add(x=step_log_probs, y=beam_state.log_probs, axis=0)
        # TODO: length penalty
        scores = log_probs
        scores = nn.reshape(scores, [-1, self.beam_size * self.vocab_size])
        # TODO: add grad for topk then this beam search can be used to train
        topk_scores, topk_indices = paddle.topk(x=scores, k=self.beam_size)
        beam_indices = nn.elementwise_floordiv(topk_indices, self.vocab_size_tensor)
        token_indices = nn.elementwise_mod(topk_indices, self.vocab_size_tensor)
        next_log_probs = self._gather(nn.reshape(log_probs, [-1, self.beam_size * self.vocab_size]), topk_indices,
                                      self.batch_size)
        next_cell_states = map_structure(lambda x: self._gather(x, beam_indices, self.batch_size), next_cell_states)
        next_finished = self._gather(beam_state.finished, beam_indices, self.batch_size)
        next_lengths = self._gather(beam_state.lengths, beam_indices, self.batch_size)
        next_lengths = next_lengths + tensor.cast(nn.logical_not(next_finished), beam_state.lengths.dtype)
        next_finished = control_flow.logical_or(next_finished, control_flow.equal(token_indices, self.end_token_tensor))

        beam_search_output = self.OutputWrapper(topk_scores, token_indices, beam_indices)
        beam_search_state = self.StateWrapper(next_cell_states, next_log_probs, next_finished, next_lengths)
        return beam_search_output, beam_search_state

    def step(self, time, inputs, states, **kwargs):
        r"""
        Perform a beam search decoding step, which uses `cell` to get probabilities,
        and follows a beam search step to calculate scores and select candidate
        token ids.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Variable): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others.
            states(Variable): A structure of tensor variables.
                It is same as the `initial_states` returned by `initialize()` for
                the first decoding step and `beam_search_state` returned by
                `step()` for the others.
            **kwargs: Additional keyword arguments, provided by the caller.

        Returns:
            tuple: A tuple( :code:`(beam_search_output, beam_search_state, next_inputs, finished)` ). \
                `beam_search_state` and `next_inputs` have the same structure, \
                shape and data type as the input arguments `states` and `inputs` separately. \
                `beam_search_output` is a namedtuple(including scores, predicted_ids, \
                parent_ids as fields) of tensor variables, where \
                `scores, predicted_ids, parent_ids` all has a tensor value shaped \
                `[batch_size, beam_size]` with data type `float32, int64, int64`. \
                `finished` is a `bool` tensor with shape `[batch_size, beam_size]`.
        """
        inputs = map_structure(self._merge_batch_beams, inputs)
        cell_states = map_structure(self._merge_batch_beams, states.cell_states)
        cell_outputs, next_cell_states = self.cell(inputs, cell_states, **kwargs)
        cell_outputs = map_structure(self._split_batch_beams, cell_outputs)
        next_cell_states = map_structure(self._split_batch_beams, next_cell_states)

        if self.output_fn is not None:
            cell_outputs = self.output_fn(cell_outputs)

        beam_search_output, beam_search_state = self._beam_search_step(time=time,
                                                                       logits=cell_outputs,
                                                                       next_cell_states=next_cell_states,
                                                                       beam_state=states)
        finished = beam_search_state.finished
        sample_ids = beam_search_output.predicted_ids
        sample_ids.stop_gradient = True
        next_inputs = self.embedding_fn(sample_ids) if self.embedding_fn else sample_ids

        return (beam_search_output, beam_search_state, next_inputs, finished)

    def finalize(self, outputs, final_states, sequence_lengths):
        r"""
        Use `gather_tree` to backtrace along the beam search tree and construct
        the full predicted sequences.

        Parameters:
            outputs(Variable): A structure(namedtuple) of tensor variables,
                The structure and data type is same as `output_dtype`.
                The tensor stacks all time steps' output thus has shape
                `[time_step, batch_size, ...]`, which is done by the caller.
            final_states(Variable): A structure(namedtuple) of tensor variables.
                It is the `next_states` returned by `decoder.step` at last
                decoding step, thus has the same structure, shape and data type
                with states at any time step.
            sequence_lengths(Variable): An `int64` tensor shaped `[batch_size, beam_size]`.
                It contains sequence lengths for each beam determined during
                decoding.

        Returns:
            tuple: A tuple( :code:`(predicted_ids, final_states)` ). \
                `predicted_ids` is an `int64` tensor shaped \
                `[time_step, batch_size, beam_size]`. `final_states` is the same \
                as the input argument `final_states`.
        """
        predicted_ids = nn.gather_tree(outputs.predicted_ids, outputs.parent_ids)
        # TODO: use FinalBeamSearchDecoderOutput as output
        return predicted_ids, final_states

    @property
    def tracks_own_finished(self):
        """
        BeamSearchDecoder reorders its beams and their finished state. Thus it
        conflicts with `dynamic_decode` function's tracking of finished states.
        Setting this property to true to avoid early stopping of decoding due
        to mismanagement of the finished state.

        Returns:
            bool: A python bool `True`.
        """
        return True


def _dynamic_decode_imperative(decoder,
                               inits=None,
                               max_step_num=None,
                               output_time_major=False,
                               impute_finished=False,
                               is_test=False,
                               return_length=False,
                               bos_ids=None,
                               **kwargs):

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = tensor.cast(state, dtype="float32")
            new_state = tensor.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = tensor.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
            step_mask.stop_gradient = True
        new_state = nn.elementwise_mul(state, step_mask, axis=0) - nn.elementwise_mul(new_state, (step_mask - 1),
                                                                                      axis=0)
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = tensor.cast(new_state, dtype=state_dtype)
        return new_state

    initial_inputs, initial_states, initial_finished = decoder.initialize(inits, bos_ids=bos_ids)
    inputs, states, finished = (initial_inputs, initial_states, initial_finished)
    cond = control_flow.logical_not((nn.reduce_all(initial_finished)))
    sequence_lengths = tensor.cast(tensor.zeros_like(initial_finished), "int64")
    outputs = None

    step_idx = 0
    step_idx_tensor = tensor.fill_constant(shape=[1], dtype="int64", value=step_idx)
    while cond.numpy():
        (step_outputs, next_states, next_inputs, next_finished) = decoder.step(step_idx_tensor, inputs, states,
                                                                               **kwargs)
        if not decoder.tracks_own_finished:
            # BeamSearchDecoder would track it own finished, since
            # beams would be reordered and the finished status of each
            # entry might change. Otherwise, perform logical OR which
            # would not change the already finished.
            next_finished = control_flow.logical_or(next_finished, finished)
            # To confirm states.finished/finished be consistent with
            # next_finished.
            tensor.assign(next_finished, finished)
            next_sequence_lengths = nn.elementwise_add(
                sequence_lengths, tensor.cast(control_flow.logical_not(finished), sequence_lengths.dtype))
            if impute_finished:  # rectify the states for the finished.
                next_states = map_structure(lambda x, y: _maybe_copy(x, y, finished), states, next_states)
        else:
            warnings.warn(
                "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
            ) if not hasattr(next_states, "lengths") else None
            next_sequence_lengths = getattr(next_states, "lengths", sequence_lengths)

        outputs = map_structure(lambda x: ArrayWrapper(x), step_outputs) if step_idx == 0 else map_structure(
            lambda x, x_array: x_array.append(x), step_outputs, outputs)
        inputs, states, finished, sequence_lengths = (next_inputs, next_states, next_finished, next_sequence_lengths)

        control_flow.increment(x=step_idx_tensor, value=1.0, in_place=True)
        step_idx += 1

        cond = control_flow.logical_not(nn.reduce_all(finished))
        if max_step_num is not None and step_idx > max_step_num:
            break

    final_outputs = map_structure(lambda x: nn.stack(x.array, axis=0), outputs)
    final_states = states

    try:
        final_outputs, final_states = decoder.finalize(final_outputs, final_states, sequence_lengths)
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = map_structure(lambda x: nn.transpose(x, [1, 0] + list(range(2, len(x.shape)))), final_outputs)

    return (final_outputs, final_states, sequence_lengths) if return_length else (final_outputs, final_states)


def _dynamic_decode_declarative(decoder,
                                inits=None,
                                max_step_num=None,
                                output_time_major=False,
                                impute_finished=False,
                                is_test=False,
                                return_length=False,
                                **kwargs):
    initial_inputs, initial_states, initial_finished = decoder.initialize(inits)
    global_inputs, global_states, global_finished = (initial_inputs, initial_states, initial_finished)
    global_finished.stop_gradient = True
    step_idx = tensor.fill_constant(shape=[1], dtype="int64", value=0)

    cond = control_flow.logical_not((nn.reduce_all(initial_finished)))
    if max_step_num is not None:
        max_step_num = tensor.fill_constant(shape=[1], dtype="int64", value=max_step_num)
    while_op = control_flow.While(cond, is_test=is_test)

    sequence_lengths = tensor.cast(tensor.zeros_like(initial_finished), "int64")
    sequence_lengths.stop_gradient = True

    if is_test:
        # for test, reuse inputs and states variables to save memory
        inputs = map_structure(lambda x: x, initial_inputs)
        states = map_structure(lambda x: x, initial_states)
    else:
        # inputs and states of all steps must be saved for backward and training
        inputs_arrays = map_structure(lambda x: control_flow.array_write(x, step_idx), initial_inputs)
        states_arrays = map_structure(lambda x: control_flow.array_write(x, step_idx), initial_states)

    def _maybe_copy(state, new_state, step_mask):
        # TODO: use where_op
        state_dtype = state.dtype
        if convert_dtype(state_dtype) in ["bool"]:
            state = tensor.cast(state, dtype="float32")
            new_state = tensor.cast(new_state, dtype="float32")
        if step_mask.dtype != state.dtype:
            step_mask = tensor.cast(step_mask, dtype=state.dtype)
            # otherwise, renamed bool gradients of would be summed up leading
            # to sum(bool) error.
            step_mask.stop_gradient = True
        new_state = nn.elementwise_mul(state, step_mask, axis=0) - nn.elementwise_mul(new_state, (step_mask - 1),
                                                                                      axis=0)
        if convert_dtype(state_dtype) in ["bool"]:
            new_state = tensor.cast(new_state, dtype=state_dtype)
        return new_state

    def _transpose_batch_time(x):
        return nn.transpose(x, [1, 0] + list(range(2, len(x.shape))))

    def _create_array_out_of_while(dtype):
        current_block_idx = default_main_program().current_block_idx
        default_main_program().current_block_idx = default_main_program().current_block().parent_idx
        tensor_array = control_flow.create_array(dtype)
        default_main_program().current_block_idx = current_block_idx
        return tensor_array

    # While
    with while_op.block():
        if not is_test:
            inputs = map_structure(lambda array: control_flow.array_read(array, step_idx), inputs_arrays)
            states = map_structure(lambda array: control_flow.array_read(array, step_idx), states_arrays)
        (outputs, next_states, next_inputs, next_finished) = decoder.step(step_idx, inputs, states, **kwargs)
        if not decoder.tracks_own_finished:
            # BeamSearchDecoder would track it own finished, since beams would
            # be reordered and the finished status of each entry might change.
            # Otherwise, perform logical OR which would not change the already
            # finished.
            next_finished = control_flow.logical_or(next_finished, global_finished)
            next_sequence_lengths = nn.elementwise_add(
                sequence_lengths, tensor.cast(control_flow.logical_not(global_finished), sequence_lengths.dtype))
            if impute_finished:  # rectify the states for the finished.
                next_states = map_structure(
                    lambda x, y: _maybe_copy(x, y, global_finished),
                    states,
                    next_states,
                )
        else:
            warnings.warn(
                "`next_states` has no `lengths` attribute, the returned `sequence_lengths` would be all zeros."
            ) if not hasattr(next_states, "lengths") else None
            next_sequence_lengths = getattr(next_states, "lengths", sequence_lengths)

        # create tensor array in global block after dtype[s] of outputs can be got
        outputs_arrays = map_structure(lambda x: _create_array_out_of_while(x.dtype), outputs)

        map_structure(lambda x, x_array: control_flow.array_write(x, i=step_idx, array=x_array), outputs,
                      outputs_arrays)
        control_flow.increment(x=step_idx, value=1.0, in_place=True)
        # update the global_finished first, since it might be also in states of
        # decoder, which otherwise would write a stale finished status to array
        tensor.assign(next_finished, global_finished)
        tensor.assign(next_sequence_lengths, sequence_lengths)
        if is_test:
            map_structure(tensor.assign, next_inputs, global_inputs)
            map_structure(tensor.assign, next_states, global_states)
        else:
            map_structure(lambda x, x_array: control_flow.array_write(x, i=step_idx, array=x_array), next_inputs,
                          inputs_arrays)
            map_structure(lambda x, x_array: control_flow.array_write(x, i=step_idx, array=x_array), next_states,
                          states_arrays)
        if max_step_num is not None:
            control_flow.logical_and(control_flow.logical_not(nn.reduce_all(global_finished)),
                                     control_flow.less_equal(step_idx, max_step_num), cond)
        else:
            control_flow.logical_not(nn.reduce_all(global_finished), cond)

    final_outputs = map_structure(lambda array: tensor.tensor_array_to_tensor(array, axis=0, use_stack=True)[0],
                                  outputs_arrays)
    if is_test:
        final_states = global_states
    else:
        final_states = map_structure(lambda array: control_flow.array_read(array, step_idx), states_arrays)

    try:
        final_outputs, final_states = decoder.finalize(final_outputs, final_states, sequence_lengths)
    except NotImplementedError:
        pass

    if not output_time_major:
        final_outputs = map_structure(_transpose_batch_time, final_outputs)

    return (final_outputs, final_states, sequence_lengths) if return_length else (final_outputs, final_states)


def dynamic_decode(decoder,
                   inits=None,
                   max_step_num=None,
                   output_time_major=False,
                   impute_finished=False,
                   is_test=False,
                   return_length=False,
                   bos_ids=None,
                   **kwargs):
    r"""
    Dynamic decoding performs :code:`decoder.step()` repeatedly until the returned
    Tensor indicating finished status contains all True values or the number of
    decoding step reaches to :attr:`max_step_num`.

    :code:`decoder.initialize()` would be called once before the decoding loop.
    If the `decoder` has implemented `finalize` method, :code:`decoder.finalize()`
    would be called once after the decoding loop.

    Parameters:
        decoder(Decoder): An instance of `Decoder`.
        inits(object, optional): Argument passed to `decoder.initialize`.
            Default `None`.
        max_step_num(int, optional): The maximum number of steps. If not provided,
            decode until the decoder is fully done, or in other words, the returned
            Tensor by :code:`decoder.step()` indicating finished status contains
            all True. Default `None`.
        output_time_major(bool, optional): Indicate the data layout of Tensor included
            in the final outputs(the first returned value of this method). If
            attr:`False`, the data layout would be batch major with shape
            `[batch_size, seq_len, ...]`.  If attr:`True`, the data layout would
            be time major with shape `[seq_len, batch_size, ...]`. Default: `False`.
        impute_finished(bool, optional): If `True` and `decoder.tracks_own_finished`
            is False, then states get copied through for batch entries which are
            marked as finished, which differs with the unfinished using the new states
            returned by :code:`decoder.step()` and ensures that the final states have
            the correct values. Otherwise, states wouldn't be copied through when
            finished. If the returned `final_states` is needed, it should be set as
            True, which causes some slowdown. Default `False`.
        is_test(bool, optional): A flag indicating whether to use test mode. In
            test mode, it is more memory saving. Default `False`.
        return_length(bool, optional):  A flag indicating whether to return an
            extra Tensor variable in the output tuple, which stores the actual
            lengths of all decoded sequences. Default `False`.
        **kwargs: Additional keyword arguments. Arguments passed to `decoder.step`.

    Returns:
        tuple: A tuple( :code:`(final_outputs, final_states, sequence_lengths)` ) \
            when `return_length` is True, otherwise a tuple( :code:`(final_outputs, final_states)` ). \
            The final outputs and states, both are Tensor or nested structure of Tensor. \
            `final_outputs` has the same structure and data types as the :code:`outputs` \
            returned by :code:`decoder.step()` , and each Tenser in `final_outputs` \
            is the stacked of all decoding steps' outputs, which might be revised \
            by :code:`decoder.finalize()` if the decoder has implemented `finalize`. \
            `final_states` is the counterpart at last time step of initial states \
            returned by :code:`decoder.initialize()` , thus has the same structure \
            with it and has tensors with same shapes and data types. `sequence_lengths` \
            is an `int64` tensor with the same shape as `finished` returned \
            by :code:`decoder.initialize()` , and it stores the actual lengths of \
            all decoded sequences.


    Examples:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle.nn import BeamSearchDecoder, dynamic_decode
            from paddle.nn import GRUCell, Linear, Embedding
            trg_embeder = Embedding(100, 32)
            output_layer = Linear(32, 32)
            decoder_cell = GRUCell(input_size=32, hidden_size=32)
            decoder = BeamSearchDecoder(decoder_cell,
                                        start_token=0,
                                        end_token=1,
                                        beam_size=4,
                                        embedding_fn=trg_embeder,
                                        output_fn=output_layer)
            encoder_output = paddle.ones((4, 8, 32), dtype=paddle.get_default_dtype())
            outputs = dynamic_decode(decoder=decoder,
                                    inits=decoder_cell.get_initial_states(encoder_output),
                                    max_step_num=10)
    """
    if in_dygraph_mode():
        return _dynamic_decode_imperative(decoder, inits, max_step_num, output_time_major, impute_finished, is_test,
                                          return_length, bos_ids, **kwargs)
    else:
        print(">>> hello_debug: not support")

        #return _dynamic_decode_declarative(decoder, inits, max_step_num,
        #                                   output_time_major, impute_finished,
        #                                  is_test, return_length, **kwargs)


class DecodeHelper(object):
    """
    DecodeHelper is the base class for any helper instance used in `BasicDecoder`.
    It provides interface to implement sampling and produce inputs for the next
    time step in dynamic decoding.
    """

    def initialize(self):
        r"""
        DecodeHelper initialization to produce inputs for the first decoding step
        and give the initial status telling whether each sequence in the batch
        is finished. It is the partial of the initialization of `BasicDecoder`.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_finished)` ). \
                `initial_inputs` is a (possibly nested structure of) tensor \
                variable[s], and the tensor's shape is `[batch_size, ...]`. \
                `initial_finished` is a bool tensor with shape `[batch_size]`.
        """
        pass

    def sample(self, time, outputs, states):
        """
        Perform sampling with some strategies according to `outputs`. It is the
        partial of `BasicDecoder.step`.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.

        Returns:
            Variable: An `int64` tensor representing the sampled ids.
        """
        pass

    def next_inputs(self, time, outputs, states, sample_ids):
        r"""
        Produce the inputs and states for next time step and give status telling
        whether each minibatch entry is finished. It is called after `sample` in
        `BasicDecoder.step`. It is the partial of `BasicDecoder.step`.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.
            sample_ids(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `sample_ids` returned by `sample()`.

        Returns:
            tuple: A tuple( :code:`(finished, next_inputs, next_states)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s], and the structure, shape and \
                data type of `next_states` must be same as the input argument \
                `states`. `finished` is a bool tensor with shape `[batch_size]`.
        """
        pass


class TrainingHelper(DecodeHelper):
    """
    TrainingHelper is a subclass of DecodeHelper. It is a decoding helper
    slicing from the full sequence inputs as the inputs for corresponding
    step. And it uses `argmax` to sample from the outputs of `cell.call()`.

    Since the needs of sequence inputs, it is used mostly for teach-forcing MLE
    (maximum likelihood) training, and the sampled would not be used.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            trg_emb = fluid.data(name="trg_emb",
                                 shape=[None, None, 128],
                                 dtype="float32")
            trg_seq_length = fluid.data(name="trg_seq_length",
                                        shape=[None],
                                        dtype="int64")
            helper = layers.TrainingHelper(trg_emb, trg_seq_length)
            decoder_cell = layers.GRUCell(hidden_size=128)
            decoder = layers.BasicDecoder(decoder_cell, helper)
            outputs = layers.dynamic_decode(
                decoder,
                inits=decoder_cell.get_initial_states(trg_emb),
                is_test=False)
    """

    def __init__(self, inputs, sequence_length, time_major=False):
        """
        Constructor of TrainingHelper.

        Parameters:
            inputs(Variable): A (possibly nested structure of) tensor variable[s].
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be sliced
                from at every decoding step.
            sequence_length(Variable): A tensor with shape `[batch_size]`.
                It stores real length of each instance in `inputs`, by which we
                can label the finished status of each instance at every decoding
                step.
            time_major(bool, optional): Indicate the data layout of Tensor included
                in `inputs`. If `False`, the data layout would be batch major with
                shape `[batch_size, sequence_length, ...]`.  If `True`, the data
                layout would be time major with shape `[sequence_length, batch_size, ...]`.
                Default: `False`.
        """
        self.inputs = inputs
        self.sequence_length = sequence_length
        self.time_major = time_major
        # extend inputs to avoid to slice out of range in `next_inputs`
        # may be easier and have better performance than condition_op
        self.inputs_ = map_structure(
            lambda x: nn.pad(x,
                             paddings=([0, 1] + [0, 0] * (len(x.shape) - 1))
                             if time_major else ([0, 0, 0, 1] + [0, 0] * (len(x.shape) - 2))), self.inputs)

    def initialize(self):
        r"""
        TrainingHelper initialization produces inputs for the first decoding
        step by slicing at the first time step of full sequence inputs, and it
        gives initial status telling whether each sequence in the batch is
        finished. It is the partial of the initialization of `BasicDecoder`.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_finished)` ). \
                `initial_inputs` is a (possibly nested structure of) tensor \
                variable[s], and the tensor's shape is `[batch_size, ...]`. \
                `initial_finished` is a bool tensor with shape `[batch_size]`.
        """
        init_finished = control_flow.equal(self.sequence_length,
                                           tensor.fill_constant(shape=[1], dtype=self.sequence_length.dtype, value=0))
        # TODO: support zero length
        init_inputs = map_structure(lambda x: x[0] if self.time_major else x[:, 0], self.inputs)
        return init_inputs, init_finished

    def sample(self, time, outputs, states):
        r"""
        Perform sampling by using `argmax` according to the `outputs`. Mostly
        the sampled ids would not be used since the inputs for next decoding
        step would be got by slicing.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the
                caller, representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.

        Returns:
            Variable: An `int64` tensor with shape `[batch_size]`, representing \
                the sampled ids.
        """
        sample_ids = tensor.argmax(outputs, axis=-1)
        return sample_ids

    def next_inputs(self, time, outputs, states, sample_ids):
        r"""
        Generate inputs for the next decoding step by slicing at corresponding
        step of the full sequence inputs. Simultaneously, produce the states
        for next time step by directly using the input `states` and emit status
        telling whether each minibatch entry reaches to the corresponding length.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the
                caller, representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.
            sample_ids(Variable): An `int64` tensor variable shaped `[batch_size]`.
                It is same as `sample_ids` returned by `sample()`.

        Returns:
            tuple: A tuple( :code:`(finished, next_inputs, next_states)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s],  and the tensor's shape is \
                `[batch_size, ...]`. `next_states` is identical to the input \
                argument `states`. `finished` is a `bool` Tensor with \
                shape `[batch_size]`.
        """
        # TODO: compatibility of int32 and int64
        time = tensor.cast(time, "int32") if convert_dtype(time.dtype) not in ["int32"] else time
        if self.sequence_length.dtype != time.dtype:
            self.sequence_length = tensor.cast(self.sequence_length, time.dtype)
        next_time = time + 1
        finished = control_flow.less_equal(self.sequence_length, next_time)

        def _slice(x):  # TODO: use Variable.__getitem__
            axes = [0 if self.time_major else 1]
            return nn.squeeze(nn.slice(x, axes=axes, starts=[next_time], ends=[next_time + 1]), axes=axes)

        next_inputs = map_structure(_slice, self.inputs_)
        return finished, next_inputs, states


class GreedyEmbeddingHelper(DecodeHelper):
    """
    GreedyEmbeddingHelper is a subclass of DecodeHelper. It is a decoding helper
    uses the argmax of the output (treated as logits) and passes the results
    through an embedding layer to get inputs for the next decoding step.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            trg_emb = fluid.data(name="trg_emb",
                                 shape=[None, None, 128],
                                 dtype="float32")

            trg_embeder = lambda x: fluid.embedding(
                x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
            output_layer = lambda x: layers.fc(x,
                                            size=10000,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name=
                                                                    "output_w"),
                                            bias_attr=False)
            helper = layers.GreedyEmbeddingHelper(trg_embeder, start_tokens=0, end_token=1)
            decoder_cell = layers.GRUCell(hidden_size=128)
            decoder = layers.BasicDecoder(decoder_cell, helper, output_fn=output_layer)
            outputs = layers.dynamic_decode(
                decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output))
    """

    def __init__(self, embedding_fn, start_tokens, end_token):
        r"""
        Constructor of GreedyEmbeddingHelper.

        Parameters:
            embedding_fn(callable): A functor to apply on the argmax results.
                Mostly it is an embedding layer to transform ids to embeddings.
                **Note that fluid.embedding should be used here rather than
                fluid.layers.embedding, since shape of ids is [batch_size].
                when using fluid.layers.embedding, must unsqueeze in embedding_fn.**
            start_tokens(Variable):  A `int64` tensor shaped `[batch_size]`,
                representing the start tokens.
            end_token(int): The end token id.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_states, finished)` ). \
                `initial_inputs` and `initial_states` both are a (possibly nested \
                structure of) tensor variable[s], and `finished` is a tensor with \
                bool data type.
        """
        self.embedding_fn = embedding_fn
        self.start_tokens = start_tokens
        self.end_token = tensor.fill_constant(shape=[1], dtype="int64", value=end_token)

    def initialize(self):
        r"""
        GreedyEmbeddingHelper initialization produces inputs for the first decoding
        step by using `start_tokens` of the constructor, and gives initial
        status telling whether each sequence in the batch is finished.
        It is the partial of the initialization of `BasicDecoder`.

        Returns:
            tuple: A tuple( :code:`(initial_inputs, initial_finished)` ). \
                `initial_inputs` is same as `start_tokens` of the constructor. \
                `initial_finished` is a `bool` tensor filled by False and has \
                the same shape as `start_tokens`.
        """
        # TODO: remove the restriction of force_cpu
        init_finished = tensor.fill_constant_batch_size_like(input=self.start_tokens,
                                                             shape=[-1],
                                                             dtype="bool",
                                                             value=False,
                                                             force_cpu=True)
        init_inputs = self.embedding_fn(self.start_tokens)
        return init_inputs, init_finished

    def sample(self, time, outputs, states):
        r"""
        Perform sampling by using `argmax` according to the `outputs`.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the
                caller, representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.

        Returns:
            Variable: An `int64` tensor with shape `[batch_size]`, representing \
                the sampled ids.
        """
        sample_ids = tensor.argmax(outputs, axis=-1)
        return sample_ids

    def next_inputs(self, time, outputs, states, sample_ids):
        r"""
        Generate inputs for the next decoding step by applying `embedding_fn`
        to `sample_ids`. Simultaneously, produce the states for next time step
        by directly using the input `states` and emit status telling whether
        each minibatch entry gets an `end_token` sample.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the
                caller, representing the current time step number of decoding.
            outputs(Variable): A tensor variable. Usually it's data type is float32
                or float64, and it's shape is `[batch_size, vocabulary_size]`,
                representing the predicted logits of current step. It is same as
                `outputs` returned by `BasicDecoder.output_fn(BasicDecoder.cell.call())`.
            states(Variable): A (possibly nested structure of) tensor variable[s].
                It is same as `new_states` returned by `BasicDecoder.cell.call()`.
            sample_ids(Variable): An `int64` tensor variable shaped `[batch_size]`.
                It is same as `sample_ids` returned by `sample()`.

        Returns:
            tuple: A tuple( :code:`(finished, next_inputs, next_states)` ). \
                `next_inputs` and `next_states` both are a (possibly nested \
                structure of) tensor variable[s],  and the tensor's shape is \
                `[batch_size, ...]`. `next_states` is identical to the input \
                argument `states`. `finished` is a `bool` Tensor with \
                shape `[batch_size]`.
        """
        finished = control_flow.equal(sample_ids, self.end_token)
        next_inputs = self.embedding_fn(sample_ids)
        return finished, next_inputs, states


class BasicDecoder(Decoder):
    """
    BasicDecoder is a subclass of Decoder and assembles a RNNCell and DecodeHelper
    instance as members, where the DecodeHelper helps to implement customed
    decoding strategies.. It performs one decoding step as following steps:

    1. Perform `cell_outputs, cell_states = cell.call(inputs, states)`
    to get outputs and new states from cell.

    2. Perform `sample_ids = helper.sample(time, cell_outputs, cell_states)`
    to sample ids as decoded results of the current time step.

    3. Perform `finished, next_inputs, next_states = helper.next_inputs(time,
    cell_outputs, cell_states, sample_ids)` to generate inputs, states and
    finished status for the next decoding step.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            trg_emb = fluid.data(name="trg_emb",
                                 shape=[None, None, 128],
                                 dtype="float32")

            trg_embeder = lambda x: fluid.embedding(
                x, size=[10000, 128], param_attr=fluid.ParamAttr(name="trg_embedding"))
            output_layer = lambda x: layers.fc(x,
                                            size=10000,
                                            num_flatten_dims=len(x.shape) - 1,
                                            param_attr=fluid.ParamAttr(name=
                                                                    "output_w"),
                                            bias_attr=False)
            helper = layers.SampleEmbeddingHelper(trg_embeder, start_tokens=0, end_token=1)
            decoder_cell = layers.GRUCell(hidden_size=128)
            decoder = layers.BasicDecoder(decoder_cell, helper, output_fn=output_layer)
            outputs = layers.dynamic_decode(
                decoder=decoder, inits=decoder_cell.get_initial_states(encoder_output))
    """

    def __init__(self, cell, helper, output_fn=None):
        """
        Constructor of BasicDecoder.

        Parameters:
            cell(RNNCell): An instance of `RNNCell` or object with the same interface.
            helper(DecodeHelper): An instance of `DecodeHelper`.
            output_fn(optional): A callable to apply to the cell's output prior to
                sampling. Default None.
        """
        self.cell = cell
        self.helper = helper
        self.output_fn = output_fn

    def initialize(self, initial_cell_states):
        r"""
        BasicDecoder initialization includes helper initialization and cell
        initialization, and cell initialization uses `initial_cell_states` as
        the result directly.

        Parameters:
            initial_cell_states(Variable): A (possibly nested structure of)
                tensor variable[s]. An argument provided by the caller `dynamic_decode`.

        Returns:
            tuple: A tuple( :code:(initial_inputs, initial_cell_states, finished)` ). \
                `initial_inputs` and `initial_states` both are a (possibly nested \
                structure of) tensor variable[s], and `finished` is a tensor with \
                bool data type. `initial_inputs` and `finished` are the results \
                of `helper.initialize()`, and `initial_cell_states` is same as \
                the input argument counterpart.
        """
        (initial_inputs, initial_finished) = self.helper.initialize()
        return initial_inputs, initial_cell_states, initial_finished

    class OutputWrapper(collections.namedtuple("OutputWrapper", ("cell_outputs", "sample_ids"))):
        """
        The structure for the returned value `outputs` of `decoder.step`.
        A namedtuple includes cell_outputs, sample_ids as fields.
        """
        pass

    def step(self, time, inputs, states, **kwargs):
        r"""
        Perform one decoding step as following steps:

        1. Perform `cell_outputs, cell_states = cell.call(inputs, states)`
        to get outputs and new states from cell.

        2. Perform `sample_ids = helper.sample(time, cell_outputs, cell_states)`
        to sample ids as decoded results of the current time step.

        3. Perform `finished, next_inputs, next_states = helper.next_inputs(time,
        cell_outputs, cell_states, sample_ids)` to generate inputs, states and
        finished status for the next decoding step.

        Parameters:
            time(Variable): An `int64` tensor with shape `[1]` provided by the caller,
                representing the current time step number of decoding.
            inputs(Variable): A tensor variable. It is same as `initial_inputs`
                returned by `initialize()` for the first decoding step and
                `next_inputs` returned by `step()` for the others.
            states(Variable): A structure of tensor variables.
                It is same as the `initial_cell_states` returned by `initialize()`
                for the first decoding step and `next_states` returned by
                `step()` for the others.
            **kwargs: Additional keyword arguments, provided by the caller
                `dynamic_decode`.

        Returns:
            tuple: A tuple( :code:`(outputs, next_states, next_inputs, finished)` ). \
                `outputs` is a namedtuple(including cell_outputs, sample_ids, \
                as fields) of tensor variables, where `cell_outputs` is the result \
                fof `cell.call()` and `sample_ids` is the result of `helper.sample()`. \
                `next_states` and `next_inputs` have the same structure, shape \
                and data type as the input arguments `states` and `inputs` separately. \
                `finished` is a `bool` tensor with shape `[batch_size]`.
        """
        cell_outputs, cell_states = self.cell(inputs, states, **kwargs)
        if self.output_fn is not None:
            cell_outputs = self.output_fn(cell_outputs)
        sample_ids = self.helper.sample(time=time, outputs=cell_outputs, states=cell_states)
        sample_ids.stop_gradient = True
        (finished, next_inputs, next_states) = self.helper.next_inputs(time=time,
                                                                       outputs=cell_outputs,
                                                                       states=cell_states,
                                                                       sample_ids=sample_ids)
        outputs = self.OutputWrapper(cell_outputs, sample_ids)
        return (outputs, next_states, next_inputs, finished)


def beam_search(pre_ids,
                pre_scores,
                ids,
                scores,
                beam_size,
                end_id,
                level=0,
                is_accumulated=True,
                name=None,
                return_parent_idx=False):
    r"""

    Beam search is a classical algorithm for selecting candidate words in a
    machine translation task.

    Refer to `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_
    for more details.

    **This operator only supports LoDTensor.** It is used after finishing
    scores calculation to perform beam search for one time step. Specifically,
    after ``ids`` and ``scores`` have been produced, it selects the top-K
    ( `k` is ``beam_size`` ) candidate word ids of current step from ``ids``
    according to the corresponding ``scores``. Additionally, ``pre_id`` and
    ``pre_scores`` are the output of `beam_search` at previous step, they
    are needed for special use to handle ended candidate translations.

    Note that if ``is_accumulated`` is True, the ``scores`` passed in should
    be accumulated scores. Otherwise, the ``scores`` are
    considered as the probabilities of single step and would be transformed to
    the log field and added up with ``pre_scores`` for final scores in this
    operator. Length penalty should be done with extra operators before calculating
    the accumulated scores if needed.

    Please see the following demo for a fully beam search usage example:

        fluid/tests/book/test_machine_translation.py

    Args:
        pre_ids(Variable): A LodTensor variable (lod level is 2), representing
            the selected ids of previous step. It is the output of beam_search
            at previous step. Its shape is `[batch_size, 1]` and its lod is
            `[[0, 1, ... , batch_size], [0, 1, ..., batch_size]]` at the
            first step. The data type should be int64.
        pre_scores(Variable): A LodTensor variable has the same shape and lod
            with ``pre_ids`` , representing the accumulated scores corresponding
            to the selected ids of previous step. It is the output of
            beam_search at previous step. The data type should be float32 or float64.
        ids(Variable|None): A LodTensor variable containing the candidates ids.
            It has the same lod with ``pre_ids`` and its shape should be
            `[batch_size * beam_size, K]`, where `K` supposed to be greater than
            ``beam_size`` and the first dimension size (decrease as samples reach
            to the end) should be same as that of ``pre_ids`` . The data type
            should be int64. It can be None, which use index in ``scores`` as
            ids.
        scores(Variable): A LodTensor variable containing the accumulated
            scores corresponding to ``ids`` . Both its shape and lod are same as
            those of ``ids`` . The data type should be float32 or float64.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        level(int): **It can be ignored and mustn't change currently.**
            The 2 level lod used in this operator has the following
            meaning: The first level describes how many beams each sample has,
            which would change to 0 when beams of the sample all end (batch reduce);
            The second level describes how many times each beam is selected.
            Default 0, which shouldn't be changed currently.
        is_accumulated(bool): Whether the input ``score`` is accumulated scores.
            Default True.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.
        return_parent_idx(bool, optional): Whether to return an extra Tensor variable
            in output, which stores the selected ids' parent index in
            ``pre_ids`` and can be used to update RNN's states by gather operator.
            Default False.

    Returns:
        tuple: The tuple contains two or three LodTensor variables. The two LodTensor, \
            representing the selected ids and the corresponding accumulated scores of \
            current step, have the same shape `[batch_size, beam_size]` and lod with 2 levels, \
            and have data types int64 and float32. If ``return_parent_idx`` is True, \
            an extra Tensor variable preserving the selected ids' parent index \
            is included, whose shape is `[batch_size * beam_size]` and data type \
            is int64.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()

            # Suppose `probs` contains predicted results from the computation
            # cell and `pre_ids` and `pre_scores` is the output of beam_search
            # at previous step.
            beam_size = 4
            end_id = 1
            pre_ids = fluid.data(
                name='pre_id', shape=[None, 1], lod_level=2, dtype='int64')
            pre_scores = fluid.data(
                name='pre_scores', shape=[None, 1], lod_level=2, dtype='float32')
            probs = fluid.data(
                name='probs', shape=[None, 10000], dtype='float32')
            topk_scores, topk_indices = fluid.layers.topk(probs, k=beam_size)
            accu_scores = fluid.layers.elementwise_add(
                x=fluid.layers.log(x=topk_scores),
                y=fluid.layers.reshape(pre_scores, shape=[-1]),
                axis=0)
            selected_ids, selected_scores = fluid.layers.beam_search(
                pre_ids=pre_ids,
                pre_scores=pre_scores,
                ids=topk_indices,
                scores=accu_scores,
                beam_size=beam_size,
                end_id=end_id)
    """
    check_variable_and_dtype(pre_ids, 'pre_ids', ['int64'], 'beam_search')
    check_variable_and_dtype(pre_scores, 'pre_scores', ['float32', 'float64'], 'beam_search')
    check_type(ids, 'ids', (Variable, type(None)), 'beam_search')
    check_variable_and_dtype(scores, 'scores', ['float32', 'float64'], 'beam_search')
    helper = LayerHelper('beam_search', **locals())
    score_type = pre_scores.dtype
    id_type = pre_ids.dtype

    inputs = {"pre_ids": pre_ids, "pre_scores": pre_scores, "scores": scores}
    if ids is not None:
        inputs["ids"] = ids

    selected_scores = helper.create_variable_for_type_inference(dtype=score_type)
    selected_ids = helper.create_variable_for_type_inference(dtype=id_type)
    # parent_idx is a tensor used to gather cell states at the next time
    # step. Though lod in selected_ids can also be used to gather by
    # sequence_expand, it is not efficient.
    # gather_op's index input only supports int32 dtype currently
    parent_idx = helper.create_variable_for_type_inference(dtype="int32")

    helper.append_op(
        type='beam_search',
        inputs=inputs,
        outputs={
            'selected_ids': selected_ids,
            'selected_scores': selected_scores,
            'parent_idx': parent_idx
        },
        attrs={
            # TODO(ChunweiYan) to assure other value support
            'level': level,
            'beam_size': beam_size,
            'end_id': end_id,
            'is_accumulated': is_accumulated,
        })
    if return_parent_idx:
        return selected_ids, selected_scores, parent_idx
    else:
        return selected_ids, selected_scores


def beam_search_decode(ids, scores, beam_size, end_id, name=None):
    r"""

    This operator is used after beam search has completed. It constructs the
    full predicted sequences for each sample by walking back along the search
    paths stored in lod of ``ids`` . The result sequences are stored in a
    LoDTensor, which uses the following way to parse:

    .. code-block:: text

        If lod = [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]

        The first level of lod stands for: There are 2 samples each having 3
        (beam width) predicted sequence.

        The second level of lod stands for: The lengths of the first sample's
        3 predicted sequences are 12, 12, 16; The lengths of the second sample's
        3 predicted sequences are 14, 13, 15.


    Please see the following demo for a fully beam search usage example:
        fluid/tests/book/test_machine_translation.py

    Args:
        ids(Variable): The LoDTensorArray variable containing the selected ids
            of all steps. Each LoDTensor in it has int64 data type and 2 level
            lod which can be used to get the search paths.
        scores(Variable): The LodTensorArray variable containing the accumulated
            scores corresponding to selected ids of all steps. It has the same size
            as ``ids`` . Each LoDTensor in it has the same shape and lod as the
            counterpart in ``ids`` , and has a float32 data type.
        beam_size(int): The beam width used in beam search.
        end_id(int): The id of end token.
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        tuple: The tuple contains two LodTensor variables. The two LodTensor, \
            containing the full sequences of ids and the corresponding accumulated \
            scores, have the same shape flattened to 1D and have the same 2 level \
            lod. The lod can be used to get how many predicted sequences each sample \
            has and how many ids each predicted sequence has.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            # Suppose `ids` and `scores` are LodTensorArray variables reserving
            # the selected ids and scores of all steps
            ids = fluid.layers.create_array(dtype='int64')
            scores = fluid.layers.create_array(dtype='float32')
            finished_ids, finished_scores = fluid.layers.beam_search_decode(
                ids, scores, beam_size=5, end_id=0)
    """
    check_variable_and_dtype(ids, 'ids', ['int64'], 'beam_search_encode')
    check_variable_and_dtype(scores, 'scores', ['float32'], 'beam_search_encode')
    helper = LayerHelper('beam_search_decode', **locals())
    sentence_ids = helper.create_variable_for_type_inference(dtype=ids.dtype)
    sentence_scores = helper.create_variable_for_type_inference(dtype=scores.dtype)

    helper.append_op(type="beam_search_decode",
                     inputs={
                         "Ids": ids,
                         "Scores": scores
                     },
                     outputs={
                         "SentenceIds": sentence_ids,
                         "SentenceScores": sentence_scores
                     },
                     attrs={
                         "beam_size": beam_size,
                         "end_id": end_id
                     })

    return sentence_ids, sentence_scores
