# coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
"""Run BERT on cmrc2018"""

import json
import os

from paddlehub.reader import tokenization
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger
from paddlehub.dataset.base_nlp_dataset import MRCDataset

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/cmrc2018.tar.gz"
SPIECE_UNDERLINE = '▁'


class CMRC2018Example(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(
            self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position is not None:
            s += ", orig_answer_text: %s" % (self.orig_answer_text)
            s += ", start_position: %d" % (self.start_position)
            s += ", end_position: %d" % (self.end_position)
        return s


class CMRC2018(MRCDataset):
    """A single set of features of data."""

    def __init__(self,
                 tokenizer=None,
                 max_seq_len=None,
                 max_query_len=64,
                 doc_stride=128):
        dataset_dir = os.path.join(DATA_HOME, "cmrc2018")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)
        super(CMRC2018, self).__init__(
            base_path=base_path,
            train_file="cmrc2018_train.json",
            dev_file="cmrc2018_dev.json",
            test_file=None,
            label_file=None,
            label_list=None,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            doc_stride=doc_stride,
        )

    def _read_file(self, input_file, phase=False):
        """Read a cmrc2018 json file into a list of CRCDExample."""

        def _is_chinese_char(cp):
            if ((cp >= 0x4E00 and cp <= 0x9FFF)
                    or (cp >= 0x3400 and cp <= 0x4DBF)
                    or (cp >= 0x20000 and cp <= 0x2A6DF)
                    or (cp >= 0x2A700 and cp <= 0x2B73F)
                    or (cp >= 0x2B740 and cp <= 0x2B81F)
                    or (cp >= 0x2B820 and cp <= 0x2CEAF)
                    or (cp >= 0xF900 and cp <= 0xFAFF)
                    or (cp >= 0x2F800 and cp <= 0x2FA1F)):
                return True
            return False

        def _is_punctuation(c):
            if c in [
                    '。', '，', '！', '？', '；', '、', '：', '（', '）', '－', '~', '「',
                    '《', '》', ',', '」', '"', '“', '”', '$', '『', '』', '—', ';',
                    '。', '(', ')', '-', '～', '。', '‘', '’', '─', ':'
            ]:
                return True
            return False

        def _tokenize_chinese_chars(text):
            """Because Chinese (and Japanese Kanji and Korean Hanja) does not have whitespace
            characters, we add spaces around every character in the CJK Unicode range before
            applying WordPiece. This means that Chinese is effectively character-tokenized.
            Note that the CJK Unicode block only includes Chinese-origin characters and
            does not include Hangul Korean or Katakana/Hiragana Japanese, which are tokenized
            with whitespace+WordPiece like all other languages."""
            output = []
            for char in text:
                cp = ord(char)
                if _is_chinese_char(cp) or _is_punctuation(char):
                    if len(output) > 0 and output[-1] != SPIECE_UNDERLINE:
                        output.append(SPIECE_UNDERLINE)
                    output.append(char)
                    output.append(SPIECE_UNDERLINE)
                else:
                    output.append(char)
            return "".join(output)

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                    c) == 0x202F or ord(c) == 0x3000 or c == SPIECE_UNDERLINE:
                return True
            return False

        examples = []
        drop = 0
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                context = _tokenize_chinese_chars(paragraph_text)

                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in context:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    if c != SPIECE_UNDERLINE:
                        char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]

                    # Only select the first answer
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    while paragraph_text[answer_offset] in [
                            " ", "\t", "\r", "\n", "。", "，", "：", ":", ".", ","
                    ]:
                        answer_offset += 1
                    start_position = char_to_word_offset[answer_offset]
                    answer_length = len(orig_answer_text)

                    end_offset = answer_offset + answer_length - 1
                    if end_offset >= len(char_to_word_offset):
                        end_offset = len(char_to_word_offset) - 1
                    end_position = char_to_word_offset[end_offset]

                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    if phase == "train":
                        actual_text = "".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = "".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            drop += 1
                            # logger.warning((actual_text, " vs ",
                            #                 cleaned_answer_text, " in ", qa))
                            continue
                    example = CMRC2018Example(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position)
                    examples.append(example)

        logger.warning("%i bad examples has been dropped" % drop)
        return examples


if __name__ == "__main__":
    print("begin")
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')
    ds = CMRC2018(tokenizer=tokenizer, max_seq_len=50)
    print("train")
    examples = ds.get_train_examples()
    for index, e in enumerate(examples):
        if index < 10:
            print(e)
    print("dev")
    examples = ds.get_dev_examples()
    for index, e in enumerate(examples):
        if index < 10:
            print(e)
