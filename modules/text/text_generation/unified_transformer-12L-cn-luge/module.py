# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
from collections import deque
from typing import List, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddlehub.module.module import moduleinfo, serving
from paddlenlp.data import Pad
from paddlenlp.transformers import UnifiedTransformerLMHeadModel, UnifiedTransformerTokenizer

from unified_transformer_12L_cn_luge.utils import select_response


@moduleinfo(
    name="unified_transformer_12L_cn_luge",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="nlp/text_generation",
)
class UnifiedTransformer(nn.Layer):
    def __init__(self):
        super(UnifiedTransformer, self).__init__()

        self.model = UnifiedTransformerLMHeadModel.from_pretrained('unified_transformer-12L-cn-luge')
        self.tokenizer = UnifiedTransformerTokenizer.from_pretrained('unified_transformer-12L-cn-luge')
        self._interactive_mode = False

    def _convert_text_to_input(self, texts: List[str], max_seq_len: int):
        """
        Convert input strings to tokens.
        """
        return self.tokenizer.dialogue_encode(
            texts, max_seq_len=max_seq_len, add_start_token_as_response=True, is_split_into_words=False)

    def _batchify(self, data: List[List[str]], max_seq_len: int, batch_size: int):
        """
        Generate input batches.
        """
        padding = False if batch_size == 1 else True
        pad_func = Pad(pad_val=self.tokenizer.pad_token_id, pad_right=False, dtype=np.int64)

        def pad_mask(batch_attention_mask):
            batch_size = len(batch_attention_mask)
            max_len = max(map(len, batch_attention_mask))
            attention_mask = np.ones((batch_size, max_len, max_len), dtype='float32') * -1e9
            for i, mask_data in enumerate(attention_mask):
                seq_len = len(batch_attention_mask[i])
                mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype='float32')
            # In order to ensure the correct broadcasting mechanism, expand one
            # dimension to the second dimension (n_head of Transformer).
            attention_mask = np.expand_dims(attention_mask, axis=1)
            return attention_mask

        def _parse_batch(batch_examples):
            if padding:
                input_ids = pad_func([example['input_ids'] for example in batch_examples])
                token_type_ids = pad_func([example['token_type_ids'] for example in batch_examples])
                position_ids = pad_func([example['position_ids'] for example in batch_examples])
                attention_mask = pad_mask([example['attention_mask'] for example in batch_examples])
            else:
                input_ids = np.asarray([example['input_ids'] for example in batch_examples], dtype=np.int64)
                token_type_ids = np.asarray([example['token_type_ids'] for example in batch_examples], dtype=np.int64)
                position_ids = np.asarray([example['position_ids'] for example in batch_examples], dtype=np.int64)
                attention_mask = np.asarray([example['attention_mask'] for example in batch_examples])
                attention_mask = np.expand_dims(attention_mask, 0)

            return input_ids, token_type_ids, position_ids, attention_mask

        examples = []
        for texts in data:
            examples.append(self._convert_text_to_input(texts, max_seq_len))

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            yield _parse_batch(one_batch)

    @contextlib.contextmanager
    def interactive_mode(self, max_turn=3):
        """
        Enter the interactive mode.
        """
        self._interactive_mode = True
        self.max_turn = max_turn
        self.context = deque(maxlen=self.max_turn)
        yield
        self.context.clear()
        self._interactive_mode = False

    def forward(self,
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                max_length=64,
                min_length=1,
                decode_strategy='sampling',
                temperature=1.0,
                top_k=5,
                top_p=1.0,
                num_beams=0,
                length_penalty=1.0,
                early_stopping=False,
                num_return_sequences=1):

        ids, scores = self.model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            min_length=min_length,
            decode_strategy=decode_strategy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_return_sequences=num_return_sequences)

        return ids, scores

    @serving
    def predict(self,
                data: Union[List[List[str]], str],
                max_seq_len: int = 512,
                batch_size: int = 1,
                use_gpu: bool = False,
                **kwargs):

        if self._interactive_mode:
            if isinstance(data, str):
                self.context.append(data.strip())
                data = [list(self.context)]
            else:
                raise ValueError("In the interactive mode, the input data should be a string.")
        elif not isinstance(data, list):
            raise ValueError("If not in the interactive mode, the input data should be a list.")

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        batches = self._batchify(data, max_seq_len, batch_size)

        results = []
        self.eval()
        for batch in batches:
            input_ids, token_type_ids, position_ids, attention_mask = map(paddle.to_tensor, batch)
            ids, scores = self(input_ids, token_type_ids, position_ids, attention_mask, **kwargs)
            num_return_sequences = 1 if 'num_return_sequences' not in kwargs\
                else kwargs['num_return_sequences']
            results.extend(
                select_response(
                    ids, scores, self.tokenizer, num_return_sequences=num_return_sequences, keep_space=False))

        if self._interactive_mode:
            self.context.append(results[0].strip())

        return results
