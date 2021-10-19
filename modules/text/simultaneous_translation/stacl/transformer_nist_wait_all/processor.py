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

import paddle
from paddlenlp.data import Vocab
from subword_nmt import subword_nmt


class STACLTokenizer:
    """
    Jieba+BPE, and convert tokens to ids.
    """

    def __init__(self,
                 bpe_codes_fpath,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 special_token=["<s>", "<e>", "<unk>"]):
        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', bpe_codes_fpath])
        bpe_args.codes.close()
        bpe_args.codes = open(bpe_codes_fpath, 'r', encoding='utf-8')
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)

        self.src_vocab = Vocab.load_vocabulary(
            src_vocab_fpath,
            bos_token=special_token[0],
            eos_token=special_token[1],
            unk_token=special_token[2])

        self.trg_vocab = Vocab.load_vocabulary(
            trg_vocab_fpath,
            bos_token=special_token[0],
            eos_token=special_token[1],
            unk_token=special_token[2])

        self.src_vocab_size = len(self.src_vocab)
        self.trg_vocab_size = len(self.trg_vocab)

    def tokenize(self, text):
        bpe_str = self.bpe.process_line(text)
        ids = self.src_vocab.to_indices(bpe_str.split())
        return bpe_str.split(), ids


def post_process_seq(seq, 
                     bos_idx=0, 
                     eos_idx=1, 
                     output_bos=False, 
                     output_eos=False):
    """
    Post-process the decoded sequence.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def predict(tokenized_src, 
              decoder_max_length, 
              is_last, 
              cache, 
              bos_id, 
              result,
              tokenizer, 
              transformer,
              n_best=1,
              max_out_len=256,
              eos_idx=1,
              waitk=-1,
    ):
    # Set evaluate mode
    transformer.eval()

    if not is_last:
        return result, cache, bos_id

    with paddle.no_grad():
        paddle.disable_static()
        input_src = tokenized_src
        if is_last:
            decoder_max_length = max_out_len
            input_src += [eos_idx]
        src_word = paddle.to_tensor(input_src).unsqueeze(axis=0)
        finished_seq, finished_scores, cache = transformer.greedy_search(
            src_word,
            max_len=decoder_max_length,
            waitk=waitk,
            caches=cache,
            bos_id=bos_id)
        finished_seq = finished_seq.numpy()
        for beam_idx, beam in enumerate(finished_seq[0]):
            if beam_idx >= n_best:
                break
            id_list = post_process_seq(beam)
            if len(id_list) == 0:
                continue
            bos_id = id_list[-1]
            word_list = tokenizer.trg_vocab.to_tokens(id_list)
            for word in word_list:
                result.append(word)
            res = ' '.join(word_list).replace('@@ ', '')
        paddle.enable_static()
    return result, cache, bos_id