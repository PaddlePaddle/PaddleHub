#coding:utf-8
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

import json
import os
import sys

from paddlehub.reader import tokenization
from paddlehub.common.downloader import default_downloader
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger

_DATA_URL = "https://bj.bcebos.com/paddlehub-dataset/squad.tar.gz"


class SquadExample(object):
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
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (tokenization.printable_text(
            self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class SQUAD(object):
    """A single set of features of data."""

    def __init__(self, version_2_with_negative=False):
        self.dataset_dir = os.path.join(DATA_HOME, "squad_data")
        if not os.path.exists(self.dataset_dir):
            ret, tips, self.dataset_dir = default_downloader.download_file_and_uncompress(
                url=_DATA_URL, save_path=DATA_HOME, print_progress=True)
        else:
            logger.info("Dataset {} already cached.".format(self.dataset_dir))
        self.version_2_with_negative = version_2_with_negative
        self._load_train_examples(version_2_with_negative, if_has_answer=True)
        self._load_dev_examples(version_2_with_negative, if_has_answer=True)

    def _load_train_examples(self,
                             version_2_with_negative=False,
                             if_has_answer=True):
        if not version_2_with_negative:
            self.train_file = os.path.join(self.dataset_dir, "train-v1.1.json")
        else:
            self.train_file = os.path.join(self.dataset_dir, "train-v2.0.json")

        self.train_examples = self._read_json(self.train_file, if_has_answer,
                                              version_2_with_negative)

    def _load_dev_examples(self,
                           version_2_with_negative=False,
                           if_has_answer=True):
        if not version_2_with_negative:
            self.dev_file = os.path.join(self.dataset_dir, "dev-v1.1.json")
        else:
            self.dev_file = os.path.join(self.dataset_dir, "dev-v2.0.json")

        self.dev_examples = self._read_json(self.dev_file, if_has_answer,
                                            version_2_with_negative)

    def _load_test_examples(self,
                            version_2_with_negative=False,
                            is_training=False):
        self.test_file = None
        logger.error("not test_file")

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return []

    def _read_json(self,
                   input_file,
                   if_has_answer,
                   version_2_with_negative=False):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r") as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(
                    c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if if_has_answer:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        # if (len(qa["answers"]) != 1) and (not is_impossible):
                        #     raise ValueError(
                        #         "For training, each question should have exactly 1 answer."
                        #     )
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[
                                answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join(
                                doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                tokenization.whitespace_tokenize(
                                    orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning(
                                    "Could not find answer: '%s' vs. '%s'" %
                                    (actual_text, cleaned_answer_text))
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)

        return examples


if __name__ == "__main__":
    ds = SQUAD(version_2_with_negative=False)
    examples = ds.get_train_examples()
    for index, e in enumerate(examples):
        if index < 10:
            print(e)
