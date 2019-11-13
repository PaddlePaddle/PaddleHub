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
"""Run BERT on DRCD"""

import json
import os
import sys

from paddlehub.reader import tokenization
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/drcd.tar.gz"
SPIECE_UNDERLINE = '▁'


class DRCDExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
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


class DRCD(object):
    """A single set of features of data."""

    def __init__(self):
        self.dataset_dir = os.path.join(DATA_HOME, "drcd")
        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))

        self._load_train_examples()
        self._load_dev_examples()
        self._load_test_examples()

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "DRCD_training.json")
        self.train_examples = self._read_json(self.train_file)

    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "DRCD_dev.json")
        self.dev_examples = self._read_json(self.dev_file)

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "DRCD_test.json")
        self.test_examples = self._read_json(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def _read_json(self, input_file):
        """Read a DRCD json file into a list of CRCDExample."""

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
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = "".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = "".join(
                        tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning((actual_text, " vs ",
                                        cleaned_answer_text, " in ", qa))
                        continue
                    example = DRCDExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position)
                    examples.append(example)
        return examples


if __name__ == "__main__":
    ds = DRCD()
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
    print("test")
    examples = ds.get_test_examples()
    for index, e in enumerate(examples):
        if index < 10:
            print(e)
