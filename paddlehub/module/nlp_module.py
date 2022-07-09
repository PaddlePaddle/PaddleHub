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
import copy
import functools
import inspect
import io
import json
import os
import six
from typing import List, Tuple, Union

import paddle
import paddle.nn as nn
from packaging.version import Version
from paddle.dataset.common import DATA_HOME
from paddle.utils.download import get_path_from_url
from paddlehub.module.module import serving, RunModule, runnable

from paddlehub.utils.log import logger
from paddlehub.utils.utils import reseg_token_label

import paddlenlp
from paddlenlp.embeddings.token_embedding import EMBEDDING_HOME, EMBEDDING_URL_ROOT
from paddlenlp.data import JiebaTokenizer
from paddlehub.compat.module.nlp_module import DataFormatError

__all__ = [
    'PretrainedModel',
    'register_base_model',
    'TransformerModule',
]


def fn_args_to_dict(func, *args, **kwargs):
    """
    Inspect function `func` and its arguments for running, and extract a
    dict mapping between argument names and keys.
    """
    if hasattr(inspect, 'getfullargspec'):
        (spec_args, spec_varargs, spec_varkw, spec_defaults, _, _, _) = inspect.getfullargspec(func)
    else:
        (spec_args, spec_varargs, spec_varkw, spec_defaults) = inspect.getargspec(func)
    # add positional argument values
    init_dict = dict(zip(spec_args, args))
    # add default argument values
    kwargs_dict = dict(zip(spec_args[-len(spec_defaults):], spec_defaults)) if spec_defaults else {}
    kwargs_dict.update(kwargs)
    init_dict.update(kwargs_dict)
    return init_dict


class InitTrackerMeta(type(nn.Layer)):
    """
    This metaclass wraps the `__init__` method of a class to add `init_config`
    attribute for instances of that class, and `init_config` use a dict to track
    the initial configuration. If the class has `_wrap_init` method, it would be
    hooked after `__init__` and called as `_wrap_init(self, init_fn, init_args)`.
    Since InitTrackerMeta would be used as metaclass for pretrained model classes,
    which always are Layer and `type(nn.Layer)` is not `type`, thus use `type(nn.Layer)`
    rather than `type` as base class for it to avoid inheritance metaclass
    conflicts.
    """

    def __init__(cls, name, bases, attrs):
        init_func = cls.__init__
        # If attrs has `__init__`, wrap it using accessable `_wrap_init`.
        # Otherwise, no need to wrap again since the super cls has been wraped.
        # TODO: remove reduplicated tracker if using super cls `__init__`
        help_func = getattr(cls, '_wrap_init', None) if '__init__' in attrs else None
        cls.__init__ = InitTrackerMeta.init_and_track_conf(init_func, help_func)
        super(InitTrackerMeta, cls).__init__(name, bases, attrs)

    @staticmethod
    def init_and_track_conf(init_func, help_func=None):
        """
        wraps `init_func` which is `__init__` method of a class to add `init_config`
        attribute for instances of that class.
        Args:
            init_func (callable): It should be the `__init__` method of a class.
            help_func (callable, optional): If provided, it would be hooked after
                `init_func` and called as `_wrap_init(self, init_func, *init_args, **init_args)`.
                Default None.
        Returns:
            function: the wrapped function
        """

        @functools.wraps(init_func)
        def __impl__(self, *args, **kwargs):
            # keep full configuration
            init_func(self, *args, **kwargs)
            # registed helper by `_wrap_init`
            if help_func:
                help_func(self, init_func, *args, **kwargs)
            self.init_config = kwargs
            if args:
                kwargs['init_args'] = args
            kwargs['init_class'] = self.__class__.__name__

        return __impl__


def register_base_model(cls):
    """
    Add a `base_model_class` attribute for the base class of decorated class,
    representing the base model class in derived classes of the same architecture.
    Args:
        cls (class): the name of the model
    """
    base_cls = cls.__bases__[0]
    assert issubclass(base_cls,
                      PretrainedModel), "`register_base_model` should be used on subclasses of PretrainedModel."
    base_cls.base_model_class = cls
    return cls


@six.add_metaclass(InitTrackerMeta)
class PretrainedModel(nn.Layer):
    """
    The base class for all pretrained models. It provides some attributes and
    common methods for all pretrained models, including attributes `init_config`,
    `config` for initialized arguments and methods for saving, loading.
    It also includes some class attributes (should be set by derived classes):
    - `model_config_file` (str): represents the file name for saving and loading
      model configuration, it's value is `model_config.json`.
    - `resource_files_names` (dict): use this to map resources to specific file
      names for saving and loading.
    - `pretrained_resource_files_map` (dict): The dict has the same keys as
      `resource_files_names`, the values are also dict mapping specific pretrained
      model name to URL linking to pretrained model.
    - `pretrained_init_configuration` (dict): The dict has pretrained model names
      as keys, and the values are also dict preserving corresponding configuration
      for model initialization.
    - `base_model_prefix` (str): represents the the attribute associated to the
      base model in derived classes of the same architecture adding layers on
      top of the base model.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {}
    # TODO: more flexible resource handle, namedtuple with fileds as:
    # resource_name, saved_file, handle_name_for_load(None for used as __init__
    # arguments), handle_name_for_save
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {}
    base_model_prefix = ""

    def _wrap_init(self, original_init, *args, **kwargs):
        """
        It would be hooked after `__init__` to add a dict including arguments of
        `__init__` as a attribute named `config` of the prtrained model instance.
        """
        init_dict = fn_args_to_dict(original_init, *args, **kwargs)
        self.config = init_dict

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)

    @property
    def model_name_list(self):
        return list(self.pretrained_init_configuration.keys())

    def get_input_embeddings(self):
        base_model = getattr(self, self.base_model_prefix, self)
        if base_model is not self:
            return base_model.get_input_embeddings()
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None  # Overwrite for models with output embeddings

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Instantiate an instance of `PretrainedModel` from a predefined
        model specified by name or path.
        Args:
            pretrained_model_name_or_path (str): A name of or a file path to a
                pretrained model.
            *args (tuple): position arguments for `__init__`. If provide, use
                this as position argument values for model initialization.
            **kwargs (dict): keyword arguments for `__init__`. If provide, use
                this to update pre-defined keyword argument values for model
                initialization.
        Returns:
            PretrainedModel: An instance of PretrainedModel.
        """
        pretrained_models = list(cls.pretrained_init_configuration.keys())
        resource_files = {}
        init_configuration = {}
        if pretrained_model_name_or_path in pretrained_models:
            for file_id, map_list in cls.pretrained_resource_files_map.items():
                resource_files[file_id] = map_list[pretrained_model_name_or_path]
            init_configuration = copy.deepcopy(cls.pretrained_init_configuration[pretrained_model_name_or_path])
        else:
            if os.path.isdir(pretrained_model_name_or_path):
                for file_id, file_name in cls.resource_files_names.items():
                    full_file_name = os.path.join(pretrained_model_name_or_path, file_name)
                    resource_files[file_id] = full_file_name
                resource_files["model_config_file"] = os.path.join(pretrained_model_name_or_path, cls.model_config_file)
            else:
                raise ValueError("Calling {}.from_pretrained() with a model identifier or the "
                                 "path to a directory instead. The supported model "
                                 "identifiers are as follows: {}".format(cls.__name__,
                                                                         cls.pretrained_init_configuration.keys()))
        # FIXME(chenzeyu01): We should use another data path for storing model
        default_root = os.path.join(DATA_HOME, pretrained_model_name_or_path)
        resolved_resource_files = {}
        for file_id, file_path in resource_files.items():
            path = os.path.join(default_root, file_path.split('/')[-1])
            if file_path is None or os.path.isfile(file_path):
                resolved_resource_files[file_id] = file_path
            elif os.path.exists(path):
                logger.info("Already cached %s" % path)
                resolved_resource_files[file_id] = path
            else:
                logger.info("Downloading %s and saved to %s" % (file_path, default_root))
                resolved_resource_files[file_id] = get_path_from_url(file_path, default_root)

        # Prepare model initialization kwargs
        # Did we saved some inputs and kwargs to reload ?
        model_config_file = resolved_resource_files.pop("model_config_file", None)
        if model_config_file is not None:
            with io.open(model_config_file, encoding="utf-8") as f:
                init_kwargs = json.load(f)
        else:
            init_kwargs = init_configuration
        # position args are stored in kwargs, maybe better not include
        init_args = init_kwargs.pop("init_args", ())
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", cls.base_model_class.__name__)

        # Check if the loaded config matches the current model class's __init__
        # arguments. If not match, the loaded config is for the base model class.
        if init_class == cls.base_model_class.__name__:
            base_args = init_args
            base_kwargs = init_kwargs
            derived_args = ()
            derived_kwargs = {}
            base_arg_index = None
        else:  # extract config for base model
            derived_args = list(init_args)
            derived_kwargs = init_kwargs
            for i, arg in enumerate(init_args):
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop("init_class") == cls.base_model_class.__name__, (
                        "pretrained base model should be {}").format(cls.base_model_class.__name__)
                    base_arg_index = i
                    break
            for arg_name, arg in init_kwargs.items():
                if isinstance(arg, dict) and "init_class" in arg:
                    assert arg.pop("init_class") == cls.base_model_class.__name__, (
                        "pretrained base model should be {}").format(cls.base_model_class.__name__)
                    base_arg_index = arg_name
                    break
            base_args = arg.pop("init_args", ())
            base_kwargs = arg
        if cls == cls.base_model_class:
            # Update with newly provided args and kwargs for base model
            base_args = base_args if not args else args
            base_kwargs.update(kwargs)
            model = cls(*base_args, **base_kwargs)
        else:
            # Update with newly provided args and kwargs for derived model
            base_model = cls.base_model_class(*base_args, **base_kwargs)
            if base_arg_index is not None:
                derived_args[base_arg_index] = base_model
            else:
                derived_args = (base_model, )  # assume at the first position
            derived_args = derived_args if not args else args
            derived_kwargs.update(kwargs)
            model = cls(*derived_args, **derived_kwargs)

        # Maybe need more ways to load resources.
        weight_path = list(resolved_resource_files.values())[0]
        assert weight_path.endswith(".pdparams"), "suffix of weight must be .pdparams"
        state_dict = paddle.load(weight_path)

        # Make sure we are able to load base models as well as derived models
        # (with heads)
        start_prefix = ""
        model_to_load = model
        state_to_load = state_dict
        unexpected_keys = []
        missing_keys = []
        if not hasattr(model, cls.base_model_prefix) and any(
                s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # base model
            state_to_load = {}
            start_prefix = cls.base_model_prefix + "."
            for k, v in state_dict.items():
                if k.startswith(cls.base_model_prefix):
                    state_to_load[k[len(start_prefix):]] = v
                else:
                    unexpected_keys.append(k)
        if hasattr(model,
                   cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            # derived model (base model with heads)
            model_to_load = getattr(model, cls.base_model_prefix)
            for k in model.state_dict().keys():
                if not k.startswith(cls.base_model_prefix):
                    missing_keys.append(k)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__,
                                                                                  unexpected_keys))
        model_to_load.set_state_dict(state_to_load)
        if paddle.in_dynamic_mode():
            return model
        return model, state_to_load

    def save_pretrained(self, save_directory):
        """
        Save model configuration and related resources (model state) to files
        under `save_directory`.
        Args:
            save_directory (str): Directory to save files into.
        """
        assert os.path.isdir(save_directory), "Saving directory ({}) should be a directory".format(save_directory)
        # save model config
        model_config_file = os.path.join(save_directory, self.model_config_file)
        model_config = self.init_config
        # If init_config contains a Layer, use the layer's init_config to save
        for key, value in model_config.items():
            if key == "init_args":
                args = []
                for arg in value:
                    args.append(arg.init_config if isinstance(arg, PretrainedModel) else arg)
                model_config[key] = tuple(args)
            elif isinstance(value, PretrainedModel):
                model_config[key] = value.init_config
        with io.open(model_config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(model_config, ensure_ascii=False))
        # save model
        file_name = os.path.join(save_directory, list(self.resource_files_names.values())[0])
        paddle.save(self.state_dict(), file_name)


class TextServing(object):
    """
    A base class for text model which supports serving.
    """

    @serving
    def predict_method(self, data: List[List[str]], max_seq_len: int = 128, batch_size: int = 1, use_gpu: bool = False):
        """
        Run predict method as a service.
        Serving as a task which is specified from serving config.
        Tasks supported:
        1. seq-cls: sequence classification;
        2. token-cls: sequence labeling;
        3. None: embedding.
        Args:
            data (obj:`List(List(str))`): The processed data whose each element is the list of a single text or a pair of texts.
            max_seq_len (:obj:`int`, `optional`, defaults to 128):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            use_gpu(obj:`bool`, defaults to `False`): Whether to use gpu to run or not.
        Returns:
            results(obj:`list`): All the predictions labels.
        """
        if self.task in self._tasks_supported:  # cls service
            if self.label_map:
                # compatible with json decoding label_map
                self.label_map = {int(k): v for k, v in self.label_map.items()}
            results = self.predict(data, max_seq_len, batch_size, use_gpu)

            if self.task == 'token-cls':
                # remove labels of [CLS] token and pad tokens
                results = [token_labels[1:len(data[i][0]) + 1] for i, token_labels in enumerate(results)]
            return results
        elif self.task is None:  # embedding service
            results = self.get_embedding(data, use_gpu)
            return results
        else:  # unknown service
            logger.error(f'Unknown task {self.task}, current tasks supported:\n'
                         '1. seq-cls: sequence classification service;\n'
                         '2. token-cls: sequence labeling service;\n'
                         '3. None: embedding service')
        return


class TransformerModule(RunModule, TextServing):
    """
    The base class for Transformer models.
    """
    _tasks_supported = [
        'seq-cls',
        'token-cls',
        'text-matching',
    ]

    @property
    def input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64')
        ]

    def _convert_text_to_input(self, tokenizer, texts: List[str], max_seq_len: int, split_char: str):
        pad_to_max_seq_len = False if self.task is None else True
        if self.task == 'token-cls':  # Extra processing of token-cls task
            tokens = texts[0].split(split_char)
            texts[0], _ = reseg_token_label(tokenizer=tokenizer, tokens=tokens)
            is_split_into_words = True
        else:
            is_split_into_words = False

        encoded_inputs = []
        if self.task == 'text-matching':
            if len(texts) != 2:
                raise RuntimeError(
                    'The input texts must have two sequences, but got %d. Please check your inputs.' % len(texts))
            encoded_inputs.append(tokenizer(text=texts[0], text_pair=None, max_seq_len=max_seq_len, \
                    pad_to_max_seq_len=True, is_split_into_words=is_split_into_words, return_length=True))
            encoded_inputs.append(tokenizer(text=texts[1], text_pair=None, max_seq_len=max_seq_len, \
                    pad_to_max_seq_len=True, is_split_into_words=is_split_into_words, return_length=True))
        else:
            if len(texts) == 1:
                if Version(paddlenlp.__version__) <= Version('2.0.0rc2'):
                    encoded_inputs.append(tokenizer.encode(texts[0], text_pair=None, \
                        max_seq_len=max_seq_len, pad_to_max_seq_len=pad_to_max_seq_len))
                else:
                    encoded_inputs.append(tokenizer(text=texts[0], max_seq_len=max_seq_len, \
                        pad_to_max_seq_len=True, is_split_into_words=is_split_into_words, return_length=True))
            elif len(texts) == 2:
                if Version(paddlenlp.__version__) <= Version('2.0.0rc2'):
                    encoded_inputs.append(tokenizer.encode(texts[0], text_pair=texts[1], \
                        max_seq_len=max_seq_len, pad_to_max_seq_len=pad_to_max_seq_len))
                else:
                    encoded_inputs.append(tokenizer(text=texts[0], text_pair=texts[1], max_seq_len=max_seq_len, \
                        pad_to_max_seq_len=True, is_split_into_words=is_split_into_words, return_length=True))
            else:
                raise RuntimeError(
                    'The input text must have one or two sequence, but got %d. Please check your inputs.' % len(texts))
        return encoded_inputs

    def _batchify(self, data: List[List[str]], max_seq_len: int, batch_size: int, split_char: str):
        def _parse_batch(batch):
            if self.task != 'text-matching':
                input_ids = [entry[0] for entry in batch]
                segment_ids = [entry[1] for entry in batch]
                return input_ids, segment_ids
            else:
                query_input_ids = [entry[0] for entry in batch]
                query_segment_ids = [entry[1] for entry in batch]
                title_input_ids = [entry[2] for entry in batch]
                title_segment_ids = [entry[3] for entry in batch]
                return query_input_ids, query_segment_ids, title_input_ids, title_segment_ids

        if not hasattr(self, 'tokenizer'):
            self.tokenizer = self.get_tokenizer()
        examples = []

        for texts in data:
            encoded_inputs = self._convert_text_to_input(self.tokenizer, texts, max_seq_len, split_char)
            example = []
            for inp in encoded_inputs:
                input_ids = inp['input_ids']
                if Version(paddlenlp.__version__) >= Version('2.0.0rc5'):
                    token_type_ids = inp['token_type_ids']
                else:
                    token_type_ids = inp['segment_ids']
                example.extend((input_ids, token_type_ids))
            examples.append(example)

        # Seperates data into some batches.
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                yield _parse_batch(one_batch)
                one_batch = []
        if one_batch:
            # The last batch whose size is less than the config batch_size setting.
            yield _parse_batch(one_batch)

    def training_step(self, batch: List[paddle.Tensor], batch_idx: int):
        """
        One step for training, which should be called as forward computation.
        Args:
            batch(:obj:List[paddle.Tensor]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.
        Returns:
            results(:obj: Dict) : The model outputs, such as loss and metrics.
        """
        if self.task == 'seq-cls':
            predictions, avg_loss, metric = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        elif self.task == 'token-cls':
            predictions, avg_loss, metric = self(
                input_ids=batch[0], token_type_ids=batch[1], seq_lengths=batch[2], labels=batch[3])
        elif self.task == 'text-matching':
            predictions, avg_loss, metric = self(query_input_ids=batch[0], query_token_type_ids=batch[1], \
                title_input_ids=batch[2], title_token_type_ids=batch[3], labels=batch[4])
        self.metric.reset()
        return {'loss': avg_loss, 'metrics': metric}

    def validation_step(self, batch: List[paddle.Tensor], batch_idx: int):
        """
        One step for validation, which should be called as forward computation.
        Args:
            batch(:obj:List[paddle.Tensor]): The one batch data, which contains the model needed,
                such as input_ids, sent_ids, pos_ids, input_mask and labels.
            batch_idx(int): The index of batch.
        Returns:
            results(:obj: Dict) : The model outputs, such as metrics.
        """
        if self.task == 'seq-cls':
            predictions, avg_loss, metric = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        elif self.task == 'token-cls':
            predictions, avg_loss, metric = self(
                input_ids=batch[0], token_type_ids=batch[1], seq_lengths=batch[2], labels=batch[3])
        elif self.task == 'text-matching':
            predictions, avg_loss, metric = self(query_input_ids=batch[0], query_token_type_ids=batch[1], \
                title_input_ids=batch[2], title_token_type_ids=batch[3], labels=batch[4])
        return {'metrics': metric}

    def get_embedding(self, data: List[List[str]], use_gpu=False):
        """
        Get token level embeddings and sentence level embeddings from model.
        Args:
            data (obj:`List(List(str))`): The processed data whose each element is the list of a single text or a pair of texts.
            use_gpu(obj:`bool`, defaults to `False`): Whether to use gpu to run or not.
        Returns:
            results(obj:`list`): All the tokens and sentences embeddings.
        """
        if self.task is not None:
            raise RuntimeError("The get_embedding method is only valid when task is None, but got task %s" % self.task)

        return self.predict(data=data, use_gpu=use_gpu)

    def predict(self,
                data: List[List[str]],
                max_seq_len: int = 128,
                split_char: str = '\002',
                batch_size: int = 1,
                use_gpu: bool = False,
                return_prob: bool = False):
        """
        Predicts the data labels.
        Args:
            data (obj:`List(List(str))`): The processed data whose each element is the list of a single text or a pair of texts.
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            split_char(obj:`str`, defaults to '\002'): The char used to split input tokens in token-cls task.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            use_gpu(obj:`bool`, defaults to `False`): Whether to use gpu to run or not.
            return_prob(obj:`bool`, defaults to `False`): Whether to return label probabilities.
        Returns:
            results(obj:`list`): All the predictions labels.
        """
        if self.task not in self._tasks_supported \
                and self.task is not None:      # None for getting embedding
            raise RuntimeError(f'Unknown task {self.task}, current tasks supported:\n'
                               '1. seq-cls: sequence classification;\n'
                               '2. token-cls: sequence labeling;\n'
                               '3. text-matching: text matching;\n'
                               '4. None: embedding')

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        batches = self._batchify(data, max_seq_len, batch_size, split_char)
        results = []
        batch_probs = []

        self.eval()
        for batch in batches:
            if self.task == 'text-matching':
                query_input_ids, query_segment_ids, title_input_ids, title_segment_ids = batch
                query_input_ids = paddle.to_tensor(query_input_ids)
                query_segment_ids = paddle.to_tensor(query_segment_ids)
                title_input_ids = paddle.to_tensor(title_input_ids)
                title_segment_ids = paddle.to_tensor(title_segment_ids)
                probs = self(query_input_ids=query_input_ids, query_token_type_ids=query_segment_ids, \
                    title_input_ids=title_input_ids, title_token_type_ids=title_segment_ids)

                idx = paddle.argmax(probs, axis=1).numpy()
                idx = idx.tolist()
                labels = [self.label_map[i] for i in idx]
            else:
                input_ids, segment_ids = batch
                input_ids = paddle.to_tensor(input_ids)
                segment_ids = paddle.to_tensor(segment_ids)
                if self.task == 'seq-cls':
                    probs = self(input_ids, segment_ids)
                    idx = paddle.argmax(probs, axis=1).numpy()
                    idx = idx.tolist()
                    labels = [self.label_map[i] for i in idx]
                elif self.task == 'token-cls':
                    probs = self(input_ids, segment_ids)
                    batch_ids = paddle.argmax(probs, axis=2).numpy()  # (batch_size, max_seq_len)
                    batch_ids = batch_ids.tolist()
                    # token labels
                    labels = [[self.label_map[i] for i in token_ids] for token_ids in batch_ids]
                elif self.task == None:
                    output = self(input_ids, segment_ids)
                    if len(output) == 1:
                        results.append(output.squeeze(0).numpy().tolist())
                    else:
                        sequence_output, pooled_output = output
                        results.append(
                            [pooled_output.squeeze(0).numpy().tolist(),
                             sequence_output.squeeze(0).numpy().tolist()])
            if self.task:
                # save probs only when return prob
                if return_prob:
                    batch_probs.extend(probs.numpy().tolist())
                results.extend(labels)

        if self.task and return_prob:
            return results, batch_probs
        return results


class EmbeddingServing(object):
    """
    A base class for embedding model which supports serving.
    """

    @serving
    def calc_similarity(self, data: List[List[str]]):
        """
        Calculate similarities of giving word pairs.
        """
        results = []
        for word_pair in data:
            if len(word_pair) != 2:
                raise RuntimeError(
                    f'The input must have two words, but got {len(word_pair)}. Please check your inputs.')
            if not isinstance(word_pair[0], str) or not isinstance(word_pair[1], str):
                raise RuntimeError(
                    f'The types of text pair must be (str, str), but got'
                    f' ({type(word_pair[0]).__name__}, {type(word_pair[1]).__name__}). Please check your inputs.')

            for word in word_pair:
                if self.get_idx_from_word(word) == \
                        self.get_idx_from_word(self.vocab.unk_token):
                    raise RuntimeError(f'Word "{word}" is not in vocab. Please check your inputs.')
            results.append(str(self.cosine_sim(*word_pair)))
        return results


class EmbeddingModule(RunModule, EmbeddingServing):
    """
    The base class for Embedding models.
    """
    base_url = 'https://paddlenlp.bj.bcebos.com/models/embeddings/'

    def _download_vocab(self):
        """
        Download vocab from url
        """
        url = EMBEDDING_URL_ROOT + '/' + f'vocab.{self.embedding_name}'
        get_path_from_url(url, EMBEDDING_HOME)

    def get_vocab_path(self):
        """
        Get local vocab path
        """
        vocab_path = os.path.join(EMBEDDING_HOME, f'vocab.{self.embedding_name}')
        if not os.path.exists(vocab_path):
            self._download_vocab()
        return vocab_path

    def get_tokenizer(self, *args, **kwargs):
        """
        Get tokenizer of embedding module
        """
        if self.embedding_name.endswith('.en'):  # English
            raise NotImplementedError  # TODO: (chenxiaojie) add tokenizer of English embedding
        else:  # Chinese
            return JiebaTokenizer(self.vocab)
