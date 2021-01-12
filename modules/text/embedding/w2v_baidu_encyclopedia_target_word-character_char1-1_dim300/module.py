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

from typing import List
from paddlenlp.embeddings import TokenEmbedding
from paddlehub.module.module import moduleinfo, serving


@moduleinfo(
    name="w2v_baidu_encyclopedia_target_word-character_char1-1_dim300",
    version="1.0.0",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="nlp/semantic_model")
class Embedding(TokenEmbedding):
    """
    Embedding model
    """
    def __init__(self, *args, **kwargs):
        super(Embedding, self).__init__(embedding_name="w2v.baidu_encyclopedia.target.word-character.char1-1.dim300", *args, **kwargs)

    @serving
    def calc_similarity(self, data: List[List[str]]):
        """
        Calculate similarities of giving word pairs.
        """
        results = []
        for word_pair in data:
            if len(word_pair) != 2:
                raise RuntimeError(
                    f'The input must have two words, but got {len(word_pair)}. Please check your inputs.')
            if not isinstance(word_pair[0], str) or not isinstance(word_pair[1], str):
                raise RuntimeError(
                    f'The types of text pair must be (str, str), but got'
                    f' ({type(word_pair[0]).__name__}, {type(word_pair[1]).__name__}). Please check your inputs.')

            for word in word_pair:
                if self.get_idx_from_word(word) == \
                        self.get_idx_from_word(self.vocab.unk_token):
                    raise RuntimeError(
                        f'Word "{word}" is not in vocab. Please check your inputs.')
            results.append(str(self.cosine_sim(*word_pair)))
        return results
