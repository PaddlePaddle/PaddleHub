#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import types
import csv
import numpy as np

#from paddlehub import dataset
from paddlehub.reader import tokenization
from paddlehub.reader.batching import prepare_batch_data


class BERTTokenizeReader(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self,
                 dataset,
                 vocab_path,
                 max_seq_len,
                 do_lower_case=True,
                 random_seed=None):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_path, do_lower_case=do_lower_case)
        self.vocab = self.tokenizer.vocab

        np.random.seed(random_seed)

        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.dataset.get_dev_examples()

    def get_val_examples(self):
        """Gets a collection of `InputExample`s for the val set."""
        return self.dataset.get_val_examples()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        return self.dataset.get_test_examples()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.dataset.get_labels()

    def convert_example(self, index, example, labels, max_seq_len, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        feature = convert_single_example(index, example, labels, max_seq_len,
                                         tokenizer)
        return feature

    def generate_instance(self, feature):
        """
        generate instance with given feature

        Args:
            feature: InputFeatures(object). A single set of features of data.
        """
        position_ids = list(range(len(feature.input_ids)))
        return [
            feature.input_ids, feature.segment_ids, position_ids,
            feature.label_id
        ]

    def generate_batch_data(self,
                            batch_data,
                            total_token_num,
                            voc_size=-1,
                            mask_id=-1,
                            return_input_mask=True,
                            return_max_len=False,
                            return_num_token=False):
        return prepare_batch_data(
            batch_data,
            total_token_num,
            voc_size=-1,
            max_seq_len=self.max_seq_len,
            pad_id=self.vocab["[PAD]"],
            cls_id=self.vocab["[CLS]"],
            sep_id=self.vocab["[SEP]"],
            mask_id=-1,
            return_input_mask=return_input_mask,
            return_max_len=return_max_len,
            return_num_token=return_num_token)

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'val', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'val'/'dev', 'test']."
            )
        return self.num_examples[phase]

    def data_generator(self, batch_size, phase='train', shuffle=True):
        """
        Generate data for train, dev/val or test.

        Args:
          batch_size: int. The batch size of generated data.
          phase: string. The phase for which to generate data.
          epoch: int. Total epoches to generate data.
          shuffle: bool. Whether to shuffle examples.
        """
        if phase == 'train':
            examples = self.get_train_examples()
            self.num_examples['train'] = len(examples)
        elif phase == 'val' or phase == 'dev':
            examples = self.get_dev_examples()
            self.num_examples['dev'] = len(examples)
        elif phase == 'test':
            examples = self.get_test_examples()
            self.num_examples['test'] = len(examples)
        else:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'dev', 'test'].")

        def instance_reader():
            """
            convert a single instance to BERT input feature
            """
            if shuffle:
                np.random.shuffle(examples)
            for (index, example) in enumerate(examples):
                feature = self.convert_example(index, example,
                                               self.get_labels(),
                                               self.max_seq_len, self.tokenizer)

                instance = self.generate_instance(feature)
                yield instance

        def batch_reader(reader, batch_size):
            batch, total_token_num, max_len = [], 0, 0
            for instance in reader():
                token_ids, sent_ids, pos_ids, label = instance[:4]
                max_len = max(max_len, len(token_ids))
                batch.append(instance)
                total_token_num += len(token_ids)
                if len(batch) == batch_size:
                    yield batch, total_token_num
                    batch, total_token_num, max_len = [], 0, 0

            if len(batch) > 0:
                yield batch, total_token_num

        def wrapper():
            for batch_data, total_token_num in batch_reader(
                    instance_reader, batch_size):
                batch_data = self.generate_batch_data(
                    batch_data,
                    total_token_num,
                    voc_size=-1,
                    mask_id=-1,
                    return_input_mask=True,
                    return_max_len=True,
                    return_num_token=False)
                yield [batch_data]

        return wrapper


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_single_example_to_unicode(guid, single_example):
    text_a = tokenization.convert_to_unicode(single_example[0])
    text_b = tokenization.convert_to_unicode(single_example[1])
    label = tokenization.convert_to_unicode(single_example[2])
    return InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


if __name__ == '__main__':
    pass
