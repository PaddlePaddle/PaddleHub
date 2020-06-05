# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import csv
import collections
from tqdm import tqdm

import numpy as np
from paddlehub.dataset import InputExample, BaseDataset
from paddlehub.common.logger import logger


class BaseNLPDataset(BaseDataset):
    def __init__(self,
                 base_path,
                 train_file=None,
                 dev_file=None,
                 test_file=None,
                 predict_file=None,
                 label_file=None,
                 label_list=None,
                 train_file_with_header=False,
                 dev_file_with_header=False,
                 test_file_with_header=False,
                 predict_file_with_header=False,
                 tokenizer=None):
        super(BaseNLPDataset, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            predict_file=predict_file,
            label_file=label_file,
            label_list=label_list,
            train_file_with_header=train_file_with_header,
            dev_file_with_header=dev_file_with_header,
            test_file_with_header=test_file_with_header,
            predict_file_with_header=predict_file_with_header)
        self.tokenizer = tokenizer
        self.train_records = self.convert_examples_to_records(
            self.train_examples, "train")
        self.dev_records = self.convert_examples_to_records(
            self.dev_examples, "dev")
        self.test_records = self.convert_examples_to_records(
            self.test_examples, "test")
        self.predict_records = self.convert_examples_to_records(
            self.predict_examples, "predict")

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        has_warned = False
        with io.open(input_file, "r", encoding="UTF-8") as file:
            reader = csv.reader(file, delimiter="\t", quotechar=None)
            examples = []
            for (i, line) in enumerate(reader):
                if i == 0:
                    ncol = len(line)
                    if self.if_file_with_header[phase]:
                        continue
                if phase != "predict":
                    if ncol == 1:
                        raise Exception(
                            "the %s file: %s only has one column but it is not a predict file"
                            % (phase, input_file))
                    elif ncol == 2:
                        example = InputExample(
                            guid=i, text_a=line[0], label=line[1])
                    elif ncol == 3:
                        example = InputExample(
                            guid=i,
                            text_a=line[0],
                            text_b=line[1],
                            label=line[2])
                    else:
                        raise Exception(
                            "the %s file: %s has too many columns (should <=3)"
                            % (phase, input_file))
                else:
                    if ncol == 1:
                        example = InputExample(guid=i, text_a=line[0])
                    elif ncol == 2:
                        if not has_warned:
                            logger.warning(
                                "the predict file: %s has 2 columns, as it is a predict file, the second one will be regarded as text_b"
                                % (input_file))
                            has_warned = True
                        example = InputExample(
                            guid=i, text_a=line[0], text_b=line[1])
                    else:
                        raise Exception(
                            "the predict file: %s has too many columns (should <=2)"
                            % (input_file))
                examples.append(example)
            return examples

    def convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, "train", "dev", "test", or "predict"

        Returns:
            a list with all the examples record.
        """
        if not self.tokenizer or not examples:
            return []

        logger.info("Processing the %s set..." % phase)

        records = []
        for example in examples:
            record = self.tokenizer.encode(
                text=example.text_a, text_pair=example.text_b)
            if example.label:
                record["label"] = self.label_list.index(example.label)
            records.append(record)
        return records

    def get_train_records(self, shuffle=False):
        records = self.train_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_dev_records(self, shuffle=False):
        records = self.dev_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_test_records(self, shuffle=False):
        records = self.test_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_val_records(self, shuffle=False):
        records = self.get_dev_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_predict_records(self, shuffle=False):
        records = self.predict_records
        if shuffle:
            np.random.shuffle(records)
        return records

    def get_records(self, phase, shuffle=False):
        if phase == "train":
            return self.get_train_records(shuffle)
        elif phase == "dev":
            return self.get_dev_records(shuffle)
        elif phase == "test":
            return self.get_test_records(shuffle)
        elif phase == "val":
            return self.get_val_records(shuffle)
        elif phase == "predict":
            return self.get_predict_records(shuffle)
        else:
            raise ValueError("Invalid phase: %s" % phase)

    def get_feed_list(self, phase):
        records = self.get_records(phase)
        if records:
            feed_list = list(records[0].keys())
        else:
            if phase == "predict":
                feed_list = [
                    feed_name for feed_name in self.get_feed_list("train")
                    if feed_name != "label"
                ]
            else:
                feed_list = [
                    feed_name for feed_name in self.get_feed_list("train")
                ]
        return feed_list


TextClassificationDataset = BaseNLPDataset


class SequenceLabelDataset(BaseNLPDataset):
    def __init__(self,
                 base_path,
                 train_file=None,
                 dev_file=None,
                 test_file=None,
                 predict_file=None,
                 label_file=None,
                 label_list=None,
                 train_file_with_header=False,
                 dev_file_with_header=False,
                 test_file_with_header=False,
                 predict_file_with_header=False,
                 tokenizer=None,
                 split_char="\002",
                 no_entity_label="O"):
        self.no_entity_label = no_entity_label
        self.split_char = split_char

        super(SequenceLabelDataset, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            predict_file=predict_file,
            label_file=label_file,
            label_list=label_list,
            train_file_with_header=train_file_with_header,
            dev_file_with_header=dev_file_with_header,
            test_file_with_header=test_file_with_header,
            predict_file_with_header=predict_file_with_header,
            tokenizer=tokenizer)

    def convert_examples_to_records(self, examples, phase):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.
            phase (str): the processing phase, "train", "dev", "test", or "predict"

        Returns:
            a list with all the examples record.
        """
        if not self.tokenizer or not examples:
            return []

        logger.info("Processing the %s set..." % phase)

        records = []
        for example in examples:
            tokens, labels = self._reseg_token_label(
                tokens=example.text_a.split(self.split_char),
                labels=example.label.split(self.split_char))
            record = self.tokenizer.encode(text=tokens)
            if labels:
                record["label"] = []
                tokens_with_specical_token = self.tokenizer.decode(
                    record, only_convert_to_tokens=True)
                tokens_index = 0
                for token in tokens_with_specical_token:
                    if tokens_index < len(
                            tokens) and token == tokens[tokens_index]:
                        record["label"].append(
                            self.label_list.index(labels[tokens_index]))
                        tokens_index += 1
                    else:
                        record["label"].append(
                            self.label_list.index(self.no_entity_label))
            records.append(record)
        return records

    def _reseg_token_label(self, tokens, labels=None):
        if labels:
            if len(tokens) != len(labels):
                raise ValueError(
                    "The length of tokens must be same with labels")
            ret_tokens = []
            ret_labels = []
            for token, label in zip(tokens, labels):
                sub_token = self.tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                ret_labels.append(label)
                if len(sub_token) < 2:
                    continue
                sub_label = label
                if label.startswith("B-"):
                    sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))

            if len(ret_tokens) != len(ret_labels):
                raise ValueError(
                    "The length of ret_tokens can't match with labels")
            return ret_tokens, ret_labels
        else:
            ret_tokens = []
            for token in tokens:
                sub_token = self.tokenizer.tokenize(token)
                if len(sub_token) == 0:
                    continue
                ret_tokens.extend(sub_token)
                if len(sub_token) < 2:
                    continue

            return ret_tokens, None


class MultiLabelDataset(BaseNLPDataset):
    def convert_examples_to_records(self, examples):
        """
        Returns a list[dict] including all the input information what the model need.

        Args:
            examples (list): the data examples, returned by _read_file.

        Returns:
            a list with all the examples record.
        """
        if not self.tokenizer:
            return []

        records = []
        for example in examples:
            record = self.tokenizer.encode(
                text=example.text_a, text_pair=example.text_b)
            if example.label:
                record["label"] = [int(label) for label in example.label]
            records.append(record)
        return records


class MRCDataset(BaseNLPDataset):
    def __init__(
            self,
            base_path,
            train_file=None,
            dev_file=None,
            test_file=None,
            predict_file=None,
            label_file=None,
            label_list=None,
            train_file_with_header=False,
            dev_file_with_header=False,
            test_file_with_header=False,
            predict_file_with_header=False,
            tokenizer=None,
            max_query_len=64,
            doc_stride=128,
    ):

        super(BaseNLPDataset, self).__init__(
            base_path=base_path,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            predict_file=predict_file,
            label_file=label_file,
            label_list=label_list,
            train_file_with_header=train_file_with_header,
            dev_file_with_header=dev_file_with_header,
            test_file_with_header=test_file_with_header,
            predict_file_with_header=predict_file_with_header,
        )

        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self._DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        self.doc_stride = doc_stride
        self._Feature = collections.namedtuple("Feature", [
            "unique_id",
            "example_index",
            "doc_span_index",
            "tokens",
            "token_to_orig_map",
            "token_is_max_context",
        ])
        self.special_tokens_num, self.special_tokens_num_before_doc = self._get_special_tokens_num(
        )

        self.train_records, self.train_features = self.convert_examples_to_records_and_features(
            self.train_examples, "train")
        self.dev_records, self.dev_features = self.convert_examples_to_records_and_features(
            self.dev_examples, "dev")
        self.test_records, self.test_features = self.convert_examples_to_records_and_features(
            self.test_examples, "test")
        self.predict_records, self.predict_features = self.convert_examples_to_records_and_features(
            self.predict_examples, "predict")

    def _get_special_tokens_num(self):
        if not self.tokenizer:
            return None, None
        # We must have a pad token, so we can use it to make fake text.
        fake_question = [self.tokenizer.pad_token]
        fake_answer = [self.tokenizer.pad_token]
        special_tokens_num = 0
        special_tokens_num_before_doc = 0
        seen_pad_num = 0
        fake_record = self.tokenizer.encode(fake_question, fake_answer)
        fake_tokens_with_special_tokens = self.tokenizer.decode(
            fake_record, only_convert_to_tokens=True)
        for token in fake_tokens_with_special_tokens:
            if token == self.tokenizer.pad_token:
                seen_pad_num += 1
                if seen_pad_num > 2:
                    # The third pad_token is added by padding
                    break
            else:
                special_tokens_num += 1
                if seen_pad_num < 2:
                    # The second pad_token is the fake_answer
                    special_tokens_num_before_doc += 1
        return special_tokens_num, special_tokens_num_before_doc

    def convert_examples_to_records_and_features(self, examples, phase):
        """Loads a data file into a list of `InputBatch`s."""
        if not self.tokenizer or not examples:
            return [], []

        logger.info("Processing the %s set..." % phase)

        features = []
        records = []
        unique_id = 1000000000

        for (example_index, example) in tqdm(enumerate(examples)):
            # Tokenize question_text
            query_tokens = self.tokenizer.tokenize(example.question_text)
            if len(query_tokens) > self.max_query_len:
                query_tokens = query_tokens[0:self.max_query_len]

            # Tokenize doc_tokens and get token-sub_token position map
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            # Update the answer position to the new sub_token position
            tok_start_position = None
            tok_end_position = None
            is_impossible = example.is_impossible if hasattr(
                example, "is_impossible") else False

            if phase != "predict" and is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if phase != "predict" and not is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position +
                                                         1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position,
                 tok_end_position) = self.improve_answer_span(
                     all_doc_tokens, tok_start_position, tok_end_position,
                     self.tokenizer, example.orig_answer_text)

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            # if hasattr(self.tokenizer, "num_special_tokens_to_add"):
            max_tokens_for_doc = self.tokenizer.max_seq_len - len(
                query_tokens) - self.special_tokens_num

            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(
                    self._DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                # Update the start_position and end_position to doc_span
                start_position = None
                end_position = None
                if phase != "predict":
                    if is_impossible:
                        start_position = 0
                        end_position = 0
                    else:
                        # For training, if our document chunk does not contain an annotation
                        # we throw it out, since there is nothing to predict.
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1
                        out_of_span = False
                        if not (tok_start_position >= doc_start
                                and tok_end_position <= doc_end):
                            out_of_span = True
                        if out_of_span:
                            start_position = 0
                            end_position = 0
                        else:
                            doc_offset = len(
                                query_tokens
                            ) + self.special_tokens_num_before_doc
                            start_position = tok_start_position - doc_start + doc_offset
                            end_position = tok_end_position - doc_start + doc_offset

                record = self.tokenizer.encode(
                    text=query_tokens,
                    text_pair=all_doc_tokens[doc_span.start:doc_span.start +
                                             doc_span.length])
                record["start_position"] = start_position
                record["end_position"] = end_position
                record["unique_id"] = unique_id
                records.append(record)

                # The other information is saved in feature, which is helpful in postprocessing.
                # The bridge with record and feature is unique_id.
                tokens = self.tokenizer.decode(
                    record, only_convert_to_tokens=True)
                token_to_orig_map = {}
                token_is_max_context = {}
                doc_token_start = len(
                    query_tokens) + self.special_tokens_num_before_doc
                for i in range(doc_span.length):
                    # split_token_index: the doc token position in doc after tokenize
                    # doc_token_index: the doc token position in record after encode
                    split_token_index = doc_span.start + i
                    doc_token_index = doc_token_start + i
                    token_to_orig_map[doc_token_index] = tok_to_orig_index[
                        split_token_index]
                    is_max_context = self.check_is_max_context(
                        doc_spans, doc_span_index, split_token_index)
                    token_is_max_context[doc_token_index] = is_max_context

                feature = self._Feature(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                )
                features.append(feature)

                unique_id += 1

        return records, features

    def improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
                            orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def check_is_max_context(self, doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context,
                        num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def get_features(self, phase):
        if phase == "train":
            return self.train_features
        elif phase == "dev":
            return self.dev_features
        elif phase == "test":
            return self.test_features
        elif phase == "val":
            return self.dev_features
        elif phase == "predict":
            return self.predict_features
        else:
            raise ValueError("Invalid phase: %s" % phase)
