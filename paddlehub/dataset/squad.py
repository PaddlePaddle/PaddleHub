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

from paddlehub.reader import tokenization
from paddlehub.common.dir import DATA_HOME
from paddlehub.common.logger import logger
from paddlehub.dataset.base_nlp_dataset import MRCDataset

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


class SQUAD(MRCDataset):
    """A single set of features of data."""

    def __init__(
            self,
            version_2_with_negative=False,
            tokenizer=None,
            max_seq_len=None,
            max_query_len=64,
            doc_stride=128,
    ):
        self.version_2_with_negative = version_2_with_negative
        if not version_2_with_negative:
            train_file = "train-v1.1.json"
            dev_file = "dev-v1.1.json"
        else:
            train_file = "train-v2.0.json"
            dev_file = "dev-v2.0.json"

        dataset_dir = os.path.join(DATA_HOME, "squad_data")
        base_path = self._download_dataset(dataset_dir, url=_DATA_URL)

        super(SQUAD, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=None,
            label_file=None,
            label_list=None,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            doc_stride=doc_stride,
        )

    def _read_file(self, input_file, phase=None):
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
                    if phase in ["train", "dev"]:
                        if self.version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if phase == "train" and (len(qa["answers"]) !=
                                                 1) and (not is_impossible):
                            print(qa)
                            raise ValueError(
                                "For training, each question should have exactly 1 answer."
                            )
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
    from paddlehub.tokenizer.bert_tokenizer import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='vocab.txt')
    ds = SQUAD(
        version_2_with_negative=True, tokenizer=tokenizer, max_seq_len=512)
    print("first 10 dev")
    for e in ds.get_dev_examples()[:2]:
        print(e)
    print("first 10 train")
    for e in ds.get_train_examples()[:2]:
        print(e)
    print("first 10 test")
    for e in ds.get_test_examples()[:2]:
        print(e)
    print(ds)
