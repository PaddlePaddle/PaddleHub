# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import csv
import io
import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import paddle
import paddlenlp
from packaging.version import Version

from paddlehub.env import DATA_HOME
from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.data import JiebaTokenizer
from paddlehub.utils.log import logger
from paddlehub.utils.utils import download, reseg_token_label, pad_sequence, trunc_sequence
from paddlehub.utils.xarfile import is_xarfile, unarchive


class InputExample(object):
    """
    The input data structure of Transformer modules (BERT, ERNIE and so on).
    """

    def __init__(self, guid: int, text_a: str, text_b: Optional[str] = None, label: Optional[str] = None):
        """
        The input data structure.
        Args:
          guid (:obj:`int`):
              Unique id for the input data.
          text_a (:obj:`str`, `optional`, defaults to :obj:`None`):
              The first sequence. For single sequence tasks, only this sequence must be specified.
          text_b (:obj:`str`, `optional`, defaults to :obj:`None`):
              The second sequence if sentence-pair.
          label (:obj:`str`, `optional`, defaults to :obj:`None`):
              The label of the example.
        Examples:
            .. code-block:: python
                from paddlehub.datasets.base_nlp_dataset import InputExample
                example = InputExample(guid=0,
                                text_a='15.4寸笔记本的键盘确实爽，基本跟台式机差不多了',
                                text_b='蛮喜欢数字小键盘，输数字特方便，样子也很美观，做工也相当不错',
                                label='1')
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return "text={}\tlabel={}".format(self.text_a, self.label)
        else:
            return "text_a={}\ttext_b={},label={}".format(self.text_a, self.text_b, self.label)


class BaseNLPDataset(object):
    """
    The virtual base class for nlp datasets, such TextClassificationDataset, SeqLabelingDataset, and so on.
    The base class must be supered and re-implemented the method _read_file.
    """

    def __init__(self,
                 base_path: str,
                 tokenizer: Union[PretrainedTokenizer, JiebaTokenizer],
                 max_seq_len: Optional[int] = 128,
                 mode: Optional[str] = "train",
                 data_file: Optional[str] = None,
                 label_file: Optional[str] = None,
                 label_list: Optional[List[str]] = None):
        """
        Ags:
            base_path (:obj:`str`): The directory to the whole dataset.
            tokenizer (:obj:`PretrainedTokenizer` or :obj:`JiebaTokenizer`):
                It tokenizes the text and encodes the data as model needed.
            max_seq_len (:obj:`int`, `optional`, defaults to :128):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).
            data_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The data file name, which is relative to the base_path.
            label_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The label file name, which is relative to the base_path.
                It is all labels of the dataset, one line one label.
            label_list(:obj:`List[str]`, `optional`, defaults to :obj:`None`):
                The list of all labels of the dataset
        """
        self.data_file = os.path.join(base_path, data_file)
        self.label_list = label_list

        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if label_file:
            self.label_file = os.path.join(base_path, label_file)
            if not self.label_list:
                self.label_list = self._load_label_data()
            else:
                logger.warning("As label_list has been assigned, label_file is noneffective")
        if self.label_list:
            self.label_map = {item: index for index, item in enumerate(self.label_list)}

    def _load_label_data(self):
        """
        Loads labels from label file.
        """
        if os.path.exists(self.label_file):
            with open(self.label_file, "r", encoding="utf8") as f:
                return f.read().strip().split("\n")
        else:
            raise RuntimeError("The file {} is not found.".format(self.label_file))

    def _download_and_uncompress_dataset(self, destination: str, url: str):
        """
        Downloads dataset and uncompresses it.
        Args:
           destination (:obj:`str`): The dataset cached directory.
           url (:obj: str): The link to be downloaded a dataset.
        """
        if not os.path.exists(destination):
            dataset_package = download(url=url, path=DATA_HOME)
            if is_xarfile(dataset_package):
                unarchive(dataset_package, DATA_HOME)
        else:
            logger.info("Dataset {} already cached.".format(destination))

    def _read_file(self, input_file: str, is_file_with_header: bool = False):
        """
        Reads the files.
        Args:
            input_file (:obj:str) : The file to be read.
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Gets all labels.
        """
        return self.label_list


class TextClassificationDataset(BaseNLPDataset, paddle.io.Dataset):
    """
    The dataset class which is fit for all datatset of text classification.
    """

    def __init__(self,
                 base_path: str,
                 tokenizer: Union[PretrainedTokenizer, JiebaTokenizer],
                 max_seq_len: int = 128,
                 mode: str = "train",
                 data_file: str = None,
                 label_file: str = None,
                 label_list: list = None,
                 is_file_with_header: bool = False):
        """
        Ags:
            base_path (:obj:`str`): The directory to the whole dataset.
            tokenizer (:obj:`PretrainedTokenizer` or :obj:`JiebaTokenizer`):
                It tokenizes the text and encodes the data as model needed.
            max_seq_len (:obj:`int`, `optional`, defaults to :128):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).
            data_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The data file name, which is relative to the base_path.
            label_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The label file name, which is relative to the base_path.
                It is all labels of the dataset, one line one label.
            label_list(:obj:`List[str]`, `optional`, defaults to :obj:`None`):
                The list of all labels of the dataset
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.
        """
        super(TextClassificationDataset, self).__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_file=label_file,
            label_list=label_list)
        self.examples = self._read_file(self.data_file, is_file_with_header)

        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file, is_file_with_header: bool = False) -> List[InputExample]:
        """
        Reads a tab separated value file.
        Args:
            input_file (:obj:str) : The file to be read.
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.
        Returns:
            examples (:obj:`List[InputExample]`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, label=line[0], text_a=line[1])
                    seq_id += 1
                    examples.append(example)
                return examples

    def _convert_examples_to_records(self, examples: List[InputExample]) -> List[dict]:
        """
        Converts all examples to records which the model needs.
        Args:
            examples(obj:`List[InputExample]`): All data examples returned by _read_file.
        Returns:
            records(:obj:`List[dict]`): All records which the model needs.
        """
        records = []
        for example in examples:
            if isinstance(self.tokenizer, PretrainedTokenizer):
                if Version(paddlenlp.__version__) <= Version('2.0.0rc2'):
                    record = self.tokenizer.encode(
                        text=example.text_a, text_pair=example.text_b, max_seq_len=self.max_seq_len)
                else:
                    record = self.tokenizer(
                        text=example.text_a,
                        text_pair=example.text_b,
                        max_seq_len=self.max_seq_len,
                        pad_to_max_seq_len=True,
                        return_length=True)
            elif isinstance(self.tokenizer, JiebaTokenizer):
                pad_token = self.tokenizer.vocab.pad_token

                ids = self.tokenizer.encode(sentence=example.text_a)
                seq_len = min(len(ids), self.max_seq_len)
                if len(ids) > self.max_seq_len:
                    ids = trunc_sequence(ids, self.max_seq_len)
                else:
                    pad_token_id = self.tokenizer.vocab.to_indices(pad_token)
                    ids = pad_sequence(ids, self.max_seq_len, pad_token_id)
                record = {'text': ids, 'seq_len': seq_len}
            else:
                raise RuntimeError(
                    "Unknown type of self.tokenizer: {}, it must be an instance of  PretrainedTokenizer or JiebaTokenizer"
                    .format(type(self.tokenizer)))

            if not record:
                logger.info(
                    "The text %s has been dropped as it has no words in the vocab after tokenization." % example.text_a)
                continue
            if example.label:
                record['label'] = self.label_map[example.label]
            records.append(record)
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        if isinstance(self.tokenizer, PretrainedTokenizer):
            input_ids = np.array(record['input_ids'])
            if Version(paddlenlp.__version__) >= Version('2.0.0rc5'):
                token_type_ids = np.array(record['token_type_ids'])
            else:
                token_type_ids = np.array(record['segment_ids'])

            if 'label' in record.keys():
                return input_ids, token_type_ids, np.array(record['label'], dtype=np.int64)
            else:
                return input_ids, token_type_ids

        elif isinstance(self.tokenizer, JiebaTokenizer):
            if 'label' in record.keys():
                return np.array(record['text']), np.array(record['label'], dtype=np.int64)
            else:
                return np.array(record['text'])
        else:
            raise RuntimeError(
                "Unknown type of self.tokenizer: {}, it must be an instance of  PretrainedTokenizer or JiebaTokenizer".
                format(type(self.tokenizer)))

    def __len__(self):
        return len(self.records)


class SeqLabelingDataset(BaseNLPDataset, paddle.io.Dataset):
    """
    Ags:
        base_path (:obj:`str`): The directory to the whole dataset.
        tokenizer (:obj:`PretrainedTokenizer` or :obj:`JiebaTokenizer`):
            It tokenizes the text and encodes the data as model needed.
        max_seq_len (:obj:`int`, `optional`, defaults to :128):
            If set to a number, will limit the total sequence returned so that it has a maximum length.
        mode (:obj:`str`, `optional`, defaults to `train`):
            It identifies the dataset mode (train, test or dev).
        data_file(:obj:`str`, `optional`, defaults to :obj:`None`):
            The data file name, which is relative to the base_path.
        label_file(:obj:`str`, `optional`, defaults to :obj:`None`):
            The label file name, which is relative to the base_path.
            It is all labels of the dataset, one line one label.
        label_list(:obj:`List[str]`, `optional`, defaults to :obj:`None`):
            The list of all labels of the dataset
        split_char(:obj:`str`, `optional`, defaults to :obj:`\002`):
            The symbol used to split chars in text and labels
        no_entity_label(:obj:`str`, `optional`, defaults to :obj:`O`):
            The label used to mark no entities
        ignore_label(:obj:`int`, `optional`, defaults to :-100):
            If one token's label == ignore_label, it will be ignored when
            calculating loss
        is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
            Whether or not the file is with the header introduction.
    """

    def __init__(self,
                 base_path: str,
                 tokenizer: Union[PretrainedTokenizer, JiebaTokenizer],
                 max_seq_len: int = 128,
                 mode: str = "train",
                 data_file: str = None,
                 label_file: str = None,
                 label_list: list = None,
                 split_char: str = "\002",
                 no_entity_label: str = "O",
                 ignore_label: int = -100,
                 is_file_with_header: bool = False):
        super(SeqLabelingDataset, self).__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_file=label_file,
            label_list=label_list)

        self.no_entity_label = no_entity_label
        self.split_char = split_char
        self.ignore_label = ignore_label

        self.examples = self._read_file(self.data_file, is_file_with_header)
        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file, is_file_with_header: bool = False) -> List[InputExample]:
        """Reads a tab separated value file."""
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, label=line[1], text_a=line[0])
                    seq_id += 1
                    examples.append(example)
                return examples

    def _convert_examples_to_records(self, examples: List[InputExample]) -> List[dict]:
        """
        Returns a list[dict] including all the input information what the model need.
        Args:
            examples (list): the data examples, returned by _read_file.
        Returns:
            a list with all the examples record.
        """
        records = []
        for example in examples:
            tokens = example.text_a.split(self.split_char)
            labels = example.label.split(self.split_char)

            # convert tokens into record
            if isinstance(self.tokenizer, PretrainedTokenizer):
                pad_token = self.tokenizer.pad_token

                tokens, labels = reseg_token_label(tokenizer=self.tokenizer, tokens=tokens, labels=labels)
                if Version(paddlenlp.__version__) <= Version('2.0.0rc2'):
                    record = self.tokenizer.encode(text=tokens, max_seq_len=self.max_seq_len)
                else:
                    record = self.tokenizer(
                        text=tokens,
                        max_seq_len=self.max_seq_len,
                        pad_to_max_seq_len=True,
                        is_split_into_words=True,
                        return_length=True)
            elif isinstance(self.tokenizer, JiebaTokenizer):
                pad_token = self.tokenizer.vocab.pad_token

                ids = [self.tokenizer.vocab.to_indices(token) for token in tokens]
                seq_len = min(len(ids), self.max_seq_len)
                if len(ids) > self.max_seq_len:
                    ids = trunc_sequence(ids, self.max_seq_len)
                else:
                    pad_token_id = self.tokenizer.vocab.to_indices(pad_token)
                    ids = pad_sequence(ids, self.max_seq_len, pad_token_id)

                record = {'text': ids, 'seq_len': seq_len}
            else:
                raise RuntimeError(
                    "Unknown type of self.tokenizer: {}, it must be an instance of  PretrainedTokenizer or JiebaTokenizer"
                    .format(type(self.tokenizer)))

            if not record:
                logger.info(
                    "The text %s has been dropped as it has no words in the vocab after tokenization." % example.text_a)
                continue

            # convert labels into record
            if labels:
                record["label"] = []
                if isinstance(self.tokenizer, PretrainedTokenizer):
                    tokens_with_specical_token = self.tokenizer.convert_ids_to_tokens(record['input_ids'])
                elif isinstance(self.tokenizer, JiebaTokenizer):
                    tokens_with_specical_token = [self.tokenizer.vocab.to_tokens(id_) for id_ in record['text']]
                else:
                    raise RuntimeError(
                        "Unknown type of self.tokenizer: {}, it must be an instance of  PretrainedTokenizer or JiebaTokenizer"
                        .format(type(self.tokenizer)))

                tokens_index = 0
                for token in tokens_with_specical_token:
                    if tokens_index < len(tokens) and token == tokens[tokens_index]:
                        record["label"].append(self.label_list.index(labels[tokens_index]))
                        tokens_index += 1
                    elif token in [pad_token]:
                        record["label"].append(self.ignore_label)  # label of special token
                    else:
                        record["label"].append(self.label_list.index(self.no_entity_label))
            records.append(record)
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        if isinstance(self.tokenizer, PretrainedTokenizer):
            input_ids = np.array(record['input_ids'])
            seq_lens = np.array(record['seq_len'])
            if Version(paddlenlp.__version__) >= Version('2.0.0rc5'):
                token_type_ids = np.array(record['token_type_ids'])
            else:
                token_type_ids = np.array(record['segment_ids'])

            if 'label' in record.keys():
                return input_ids, token_type_ids, seq_lens, np.array(record['label'], dtype=np.int64)
            else:
                return input_ids, token_type_ids, seq_lens

        elif isinstance(self.tokenizer, JiebaTokenizer):
            if 'label' in record.keys():
                return np.array(record['text']), np.array(record['seq_len']), np.array(record['label'], dtype=np.int64)
            else:
                return np.array(record['text']), np.array(record['seq_len'])
        else:
            raise RuntimeError(
                "Unknown type of self.tokenizer: {}, it must be an instance of  PretrainedTokenizer or JiebaTokenizer".
                format(type(self.tokenizer)))

    def __len__(self):
        return len(self.records)


class TextMatchingDataset(BaseNLPDataset, paddle.io.Dataset):
    """
    The dataset class which is fit for all datatset of text matching.
    """

    def __init__(self,
                 base_path: str,
                 tokenizer: PretrainedTokenizer,
                 max_seq_len: int = 128,
                 mode: str = "train",
                 data_file: str = None,
                 label_file: str = None,
                 label_list: list = None,
                 is_file_with_header: bool = False):
        """
        Ags:
            base_path (:obj:`str`): The directory to the whole dataset.
            tokenizer (:obj:`PretrainedTokenizer`):
                It tokenizes the text and encodes the data as model needed.
            max_seq_len (:obj:`int`, `optional`, defaults to :128):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            mode (:obj:`str`, `optional`, defaults to `train`):
                It identifies the dataset mode (train, test or dev).
            data_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The data file name, which is relative to the base_path.
            label_file(:obj:`str`, `optional`, defaults to :obj:`None`):
                The label file name, which is relative to the base_path.
                It is all labels of the dataset, one line one label.
            label_list(:obj:`List[str]`, `optional`, defaults to :obj:`None`):
                The list of all labels of the dataset
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.
        """
        super(TextMatchingDataset, self).__init__(
            base_path=base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_file=label_file,
            label_list=label_list)
        self.examples = self._read_file(self.data_file, is_file_with_header)

        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file, is_file_with_header: bool = False) -> List[InputExample]:
        """
        Reads a tab separated value file.
        Args:
            input_file (:obj:str) : The file to be read.
            is_file_with_header(:obj:bool, `optional`, default to :obj: False) :
                Whether or not the file is with the header introduction.
        Returns:
            examples (:obj:`List[InputExample]`): All the input data.
        """
        if not os.path.exists(input_file):
            raise RuntimeError("The file {} is not found.".format(input_file))
        else:
            with io.open(input_file, "r", encoding="UTF-8") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                examples = []
                seq_id = 0
                header = next(reader) if is_file_with_header else None
                for line in reader:
                    example = InputExample(guid=seq_id, text_a=line[0], text_b=line[1], label=line[2])
                    seq_id += 1
                    examples.append(example)
                return examples

    def _convert_examples_to_records(self, examples: List[InputExample]) -> List[dict]:
        """
        Converts all examples to records which the model needs.
        Args:
            examples(obj:`List[InputExample]`): All data examples returned by _read_file.
        Returns:
            records(:obj:`List[dict]`): All records which the model needs.
        """
        records = []
        for example in examples:
            if isinstance(self.tokenizer, PretrainedTokenizer):
                record_a = self.tokenizer(text=example.text_a, max_seq_len=self.max_seq_len, \
                    pad_to_max_seq_len=True, return_length=True)
                record_b = self.tokenizer(text=example.text_b, max_seq_len=self.max_seq_len, \
                    pad_to_max_seq_len=True, return_length=True)
                record = {'text_a': record_a, 'text_b': record_b}
            else:
                raise RuntimeError(
                    "Unknown type of self.tokenizer: {}, it must be an instance of PretrainedTokenizer".format(
                        type(self.tokenizer)))

            if not record:
                logger.info(
                    "The text %s has been dropped as it has no words in the vocab after tokenization." % example.text_a)
                continue
            if example.label:
                record['label'] = self.label_map[example.label]
            records.append(record)
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        if isinstance(self.tokenizer, PretrainedTokenizer):
            query_input_ids = np.array(record['text_a']['input_ids'])
            query_token_type_ids = np.array(record['text_a']['token_type_ids'])
            title_input_ids = np.array(record['text_b']['input_ids'])
            title_token_type_ids = np.array(record['text_b']['token_type_ids'])

            if 'label' in record.keys():
                return query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids, \
                    np.array(record['label'], dtype=np.int64)
            else:
                return query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids
        else:
            raise RuntimeError(
                "Unknown type of self.tokenizer: {}, it must be an instance of PretrainedTokenizer".format(
                    type(self.tokenizer)))

    def __len__(self):
        return len(self.records)
