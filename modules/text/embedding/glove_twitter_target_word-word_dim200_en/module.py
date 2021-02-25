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

from paddlenlp.embeddings import TokenEmbedding
from paddlehub.module.module import moduleinfo
from paddlehub.module.nlp_module import EmbeddingModule


@moduleinfo(
    name="glove_twitter_target_word-word_dim200_en",
    version="1.0.1",
    summary="",
    author="paddlepaddle",
    author_email="",
    type="nlp/semantic_model",
    meta=EmbeddingModule)
class Embedding(TokenEmbedding):
    """
    Embedding model
    """
    embedding_name = "glove.twitter.target.word-word.dim200.en"

    def __init__(self, *args, **kwargs):
        super(Embedding, self).__init__(embedding_name=self.embedding_name, *args, **kwargs)
