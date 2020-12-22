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

# FIXME(zhangxuefei): remove this file after paddlenlp is released.

import copy
import functools
import inspect
import io
import json
import os
import six
from typing import List

import paddle
import paddle.nn as nn
from paddle.dataset.common import DATA_HOME
from paddle.utils.download import get_path_from_url
from paddlehub.module.module import serving, RunModule, runnable

from paddlehub.utils.log import logger

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


class EmbeddingServing(object):
    @serving
    def get_embedding(self, texts, use_gpu=False):
        if self.task is not None:
            raise RuntimeError("The get_embedding method is only valid when task is None, but got task %s" % self.task)

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')

        tokenizer = self.get_tokenizer()
        results = []
        for text in texts:
            if len(text) == 1:
                encoded_inputs = tokenizer.encode(text[0], text_pair=None, pad_to_max_seq_len=False)
            elif len(text) == 2:
                encoded_inputs = tokenizer.encode(text[0], text_pair=text[1], pad_to_max_seq_len=False)
            else:
                raise RuntimeError(
                    'The input text must have one or two sequence, but got %d. Please check your inputs.' % len(text))

            input_ids = paddle.to_tensor(encoded_inputs['input_ids']).unsqueeze(0)
            segment_ids = paddle.to_tensor(encoded_inputs['segment_ids']).unsqueeze(0)
            sequence_output, pooled_output = self(input_ids, segment_ids)

            sequence_output = sequence_output.squeeze(0)
            pooled_output = pooled_output.squeeze(0)
            results.append((sequence_output.numpy().tolist(), pooled_output.numpy().tolist()))
        return results


class TransformerModule(RunModule, EmbeddingServing):
    _tasks_supported = [
        'seq-cls',
        'token-cls',
    ]

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
        predictions, avg_loss, acc = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        return {'loss': avg_loss, 'metrics': {'acc': acc}}

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
        predictions, avg_loss, acc = self(input_ids=batch[0], token_type_ids=batch[1], labels=batch[2])
        return {'metrics': {'acc': acc}}

    def predict(self, data, max_seq_len=128, batch_size=1, use_gpu=False):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(str)`): The processed data whose each element is the raw text.
            max_seq_len (:obj:`int`, `optional`, defaults to :int:`None`):
                If set to a number, will limit the total sequence returned so that it has a maximum length.
            batch_size(obj:`int`, defaults to 1): The number of batch.
            use_gpu(obj:`bool`, defaults to `False`): Whether to use gpu to run or not.

        Returns:
            results(obj:`list`): All the predictions labels.
        """
        if self.task not in self._tasks_supported:
            raise RuntimeError("The predict method supports task in {}, but got task {}.".format(
                self._tasks_supported, self.task))

        paddle.set_device('gpu') if use_gpu else paddle.set_device('cpu')
        tokenizer = self.get_tokenizer()

        examples = []
        for text in data:
            if len(text) == 1:
                encoded_inputs = tokenizer.encode(text[0], text_pair=None, max_seq_len=max_seq_len)
            elif len(text) == 2:
                encoded_inputs = tokenizer.encode(text[0], text_pair=text[1], max_seq_len=max_seq_len)
            else:
                raise RuntimeError(
                    'The input text must have one or two sequence, but got %d. Please check your inputs.' % len(text))
            examples.append((encoded_inputs['input_ids'], encoded_inputs['segment_ids']))

        def _batchify_fn(batch):
            input_ids = [entry[0] for entry in batch]
            segment_ids = [entry[1] for entry in batch]
            return input_ids, segment_ids

        # Seperates data into some batches.
        batches = []
        one_batch = []
        for example in examples:
            one_batch.append(example)
            if len(one_batch) == batch_size:
                batches.append(one_batch)
                one_batch = []
        if one_batch:
            # The last batch whose size is less than the config batch_size setting.
            batches.append(one_batch)

        results = []
        self.eval()
        for batch in batches:
            input_ids, segment_ids = _batchify_fn(batch)
            input_ids = paddle.to_tensor(input_ids)
            segment_ids = paddle.to_tensor(segment_ids)

            if self.task == 'seq-cls':
                probs = self(input_ids, segment_ids)
                idx = paddle.argmax(probs, axis=1).numpy()
                idx = idx.tolist()
                labels = [self.label_map[i] for i in idx]
                results.extend(labels)
            elif self.task == 'token-cls':
                probs = self(input_ids, segment_ids)
                batch_ids = paddle.argmax(probs, axis=2).numpy()  # (batch_size, max_seq_len)
                batch_ids = batch_ids.tolist()
                token_labels = [[self.label_map[i] for i in token_ids] for token_ids in batch_ids]
                results.extend(token_labels)

        return results
