# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


def post_process_response(token_ids: List[int], tokenizer):
    '''
    Post-process the decoded sequence. Truncate from the first <eos>.
    '''
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return token_ids, tokens


def get_in_turn_repetition(pred: List[str], is_cn: bool = True):
    '''
    Get in-turn repetition.
    '''
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def select_response(ids,
                    scores: List[float],
                    tokenizer,
                    max_dec_len: int = None,
                    num_samples: int = 1,
                    is_cn: bool = True):
    '''
    Select response with the highest score.
    '''
    ids = ids.numpy().tolist()
    scores = scores.numpy()

    if len(ids) != len(scores) or (len(ids) % num_samples) != 0:
        raise ValueError("the length of `ids` is {}, but the `num_samples` is {}".format(len(ids), num_samples))

    group = []
    tmp = []
    for pred, score in zip(ids, scores):
        pred_token_ids, pred_tokens = post_process_response(pred, tokenizer)
        num_token = len(pred_token_ids)
        response = " ".join(pred_tokens)

        in_turn_repetition = get_in_turn_repetition(pred_tokens) or get_in_turn_repetition(pred_token_ids, False)
        # not ending
        if max_dec_len is not None and num_token >= max_dec_len:
            score -= 1e3
        elif in_turn_repetition:
            score -= 1e3

        tmp.append([response, score])
        if len(tmp) == num_samples:
            group.append(tmp)
            tmp = []

    results = []
    for preds in group:
        preds = sorted(preds, key=lambda x: -x[1])
        results.append(preds[0][0].replace(' ', '') if is_cn else preds[0][0])
    return results
