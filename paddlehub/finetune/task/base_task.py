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

import os
import contextlib
import time
import copy
import inspect
from functools import partial
from collections import OrderedDict
import six
if six.PY2:
    from inspect import getargspec as get_args
else:
    from inspect import getfullargspec as get_args
import numpy as np
import paddle.fluid as fluid
from tb_paddle import SummaryWriter

import paddlehub as hub
from paddlehub.common.paddle_helper import dtype_map, clone_program
from paddlehub.common.utils import mkdir
from paddlehub.common.dir import tmp_dir
from paddlehub.common.logger import logger
from paddlehub.finetune.checkpoint import load_checkpoint, save_checkpoint
from paddlehub.finetune.config import RunConfig


class RunState(object):
    """
    RunState is used to save the result of every running step

    Args:
        length (int): the number of fetch result
    """

    def __init__(self, length):
        self.run_time_begin = time.time()
        self.run_step = 0
        self.run_examples = 0
        self.run_results = [0] * length
        self.run_time_used = 0
        self.run_speed = 0.0

    def __add__(self, other):
        self.run_step += other.run_step
        self.run_examples += other.run_examples
        for index in range(len(self.run_results)):
            self.run_results[index] += other.run_results[index]
        return self

    def update(self):
        self.run_time_used = time.time() - self.run_time_begin
        self.run_speed = self.run_step / self.run_time_used
        return self


class RunEnv(object):
    """
    RunEnv saves the running environment of the train/dev/predict phase, including program, reader, metrics and so on.
    """

    def __init__(self):
        self.current_epoch = 0
        self.current_step = 0
        self.main_program = None
        self.start_program = None
        self.main_program_compiled = None
        self.py_reader = None
        self.reader = None
        self.loss = None
        self.labels = None
        self.metrics = None
        self.is_inititalized = False
        self.UNG = copy.deepcopy(fluid.unique_name.generator)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getattr__(self, key):
        return self.__dict__[key]


class TaskHooks():
    """
    TaskHooks can handle some tasks during the spectific event.
    """

    def __init__(self):
        self._registered_hooks = {
            "build_env_start_event": OrderedDict(),
            "build_env_end_event": OrderedDict(),
            "finetune_start_event": OrderedDict(),
            "finetune_end_event": OrderedDict(),
            "predict_start_event": OrderedDict(),
            "predict_end_event": OrderedDict(),
            "eval_start_event": OrderedDict(),
            "eval_end_event": OrderedDict(),
            "log_interval_event": OrderedDict(),
            "save_ckpt_interval_event": OrderedDict(),
            "eval_interval_event": OrderedDict(),
            "run_step_event": OrderedDict(),
        }
        self._hook_params_num = {
            "build_env_start_event": 1,
            "build_env_end_event": 1,
            "finetune_start_event": 1,
            "finetune_end_event": 2,
            "predict_start_event": 1,
            "predict_end_event": 2,
            "eval_start_event": 1,
            "eval_end_event": 2,
            "log_interval_event": 2,
            "save_ckpt_interval_event": 1,
            "eval_interval_event": 1,
            "run_step_event": 2,
        }

    def add(self, hook_type, name=None, func=None):
        """
        add the handler function to spectific event.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name, default None
            func (func): the handler function, default None
        """
        if not func or not callable(func):
            raise TypeError(
                "The hook function is empty or it is not a function")
        if name == None:
            name = "hook_%s" % id(func)

        # check validity
        if not isinstance(name, str) or name.strip() == "":
            raise TypeError("The hook name must be a non-empty string")
        if hook_type not in self._registered_hooks:
            raise ValueError("hook_type: %s does not exist" % (hook_type))
        if name in self._registered_hooks[hook_type]:
            raise ValueError(
                "name: %s has existed in hook_type:%s, use modify method to modify it"
                % (name, hook_type))
        else:
            args_num = len(get_args(func).args)
            if args_num != self._hook_params_num[hook_type]:
                raise ValueError(
                    "The number of parameters to the hook hook_type:%s should be %i"
                    % (hook_type, self._hook_params_num[hook_type]))
            self._registered_hooks[hook_type][name] = func

    def delete(self, hook_type, name):
        """
        delete the handler function of spectific event.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
        """
        if self.exist(hook_type, name):
            del self._registered_hooks[hook_type][name]
        else:
            raise ValueError(
                "No hook_type: %s exists or name: %s does not exist in hook_type: %s"
                % (hook_type, name, hook_type))

    def modify(self, hook_type, name, func):
        """
        modify the handler function of spectific event.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
            func (func): the new handler function
        """
        if not (isinstance(name, str) and callable(func)):
            raise TypeError(
                "The hook name must be a string, and the hook function must be a function"
            )
        if self.exist(hook_type, name):
            self._registered_hooks[hook_type][name] = func
        else:
            raise ValueError(
                "No hook_type: %s exists or name: %s does not exist in hook_type: %s"
                % (hook_type, name, hook_type))

    def exist(self, hook_type, name):
        """
        check if the the handler function of spectific event is existing.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name

        Returns:
            bool: True or False
        """
        if hook_type not in self._registered_hooks \
                or name not in self._registered_hooks[hook_type]:
            return False
        else:
            return True

    def info(self, show_default=False):
        """
        get the hooks information, including the source code.

        Args:
            show_default (bool): show the information of Paddlehub default hooks or not, default False

        Returns:
            str: the formatted string of the hooks information
        """
        # formatted output the source code
        ret = ""
        for hook_type, hooks in self._registered_hooks.items():
            already_print_type = False
            for name, func in hooks.items():
                if name == "default" and not show_default:
                    continue
                if not already_print_type:
                    ret += "hook_type: %s{\n" % hook_type
                    already_print_type = True
                source = inspect.getsource(func)
                ret += " name: %s{\n" % name
                for line in source.split("\n"):
                    ret += "  %s\n" % line
                ret += " }\n"
            if already_print_type:
                ret += "}\n"
        if not ret:
            ret = "Not any customized hooks have been defined, you can set show_default=True to see the default hooks information"
        return ret

    def __getitem__(self, hook_type):
        return self._registered_hooks[hook_type]

    def __repr__(self):
        return self.info(show_default=False)


class BaseTask(object):
    """
    BaseTask is the base class of all the task. It will complete the building of all the running environment.

    Args:
        feed_list (list): the inputs name
        data_reader (object): data reader for the task
        main_program (object): the customized main_program, default None
        startup_program (object): the customized startup_program, default None
        config (object): the config for the task, default None
        metrics_choices (list): metrics used to the task, default ["acc"]
    """

    def __init__(self,
                 feed_list,
                 data_reader,
                 main_program=None,
                 startup_program=None,
                 config=None,
                 metrics_choices="default"):
        # base item
        self._base_data_reader = data_reader
        self._base_feed_list = feed_list

        # metrics item
        self.best_score = -999
        if metrics_choices == "default":
            metrics_choices = ["acc"]
        elif metrics_choices == None:
            metrics_choices = []
        if isinstance(metrics_choices, list):
            self.metrics_choices = metrics_choices
        else:
            self.metrics_choices = [metrics_choices]

        if main_program is None:
            self._base_main_program = clone_program(
                fluid.default_main_program(), for_test=False)

        else:
            self._base_main_program = clone_program(
                main_program, for_test=False)
        if startup_program is None:
            self._base_startup_program = clone_program(
                fluid.default_startup_program(), for_test=False)
        else:
            self._base_startup_program = clone_program(
                startup_program, for_test=False)
        self.is_checkpoint_loaded = False
        self._base_compiled_program = None

        # run config
        self.config = config if config else RunConfig()
        self.place = self.places[0]
        self.device_count = len(self.places)

        if self.config.use_data_parallel:
            if not self.config.use_pyreader and self.config.batch_size < self.device_count:
                logger.warning(
                    "Batch size({}) is less than the count of devices({}), which is not allowed in current Paddle versions"
                    .format(self.config.batch_size, self.device_count))
                logger.warning("Batch size automatically adjusted to {}".format(
                    self.device_count))
                self.config._batch_size = self.device_count

        self.exe = fluid.Executor(place=self.place)
        self.build_strategy = fluid.BuildStrategy()

        # run environment
        self._phases = []
        self._envs = {}
        self._predict_data = None
        self._tb_writer = None

        # event hooks
        self._hooks = TaskHooks()
        for hook_type, event_hooks in self._hooks._registered_hooks.items():
            self._hooks.add(hook_type, "default",
                            eval("self._default_%s" % hook_type))
            setattr(BaseTask, "_%s" % hook_type,
                    self.create_event_function(hook_type))

        # accelerate predict
        self.is_best_model_loaded = False
        self._predictor = None

        # set default phase
        self.enter_phase("train")

    @contextlib.contextmanager
    def phase_guard(self, phase):
        self.enter_phase(phase)
        yield
        self.exit_phase()

    def enter_phase(self, phase):
        if phase not in ["train", "val", "dev", "test", "predict", "inference"]:
            raise RuntimeError()
        if phase in ["val", "dev"]:
            phase = "dev"
        elif phase in ["predict", "inference"]:
            phase = "predict"
        self._phases.append(phase)

    def exit_phase(self):
        self._phases = self._phases[:-1]

    def init_if_necessary(self):
        if not self.is_checkpoint_loaded:
            if not self.load_checkpoint():
                self.exe.run(self._base_startup_program)
            self.is_checkpoint_loaded = True
            self.is_best_model_loaded = False

    def init_if_load_best_model(self):
        if not self.is_best_model_loaded:
            best_model_path = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            logger.info("Load the best model from %s" % best_model_path)
            if os.path.exists(best_model_path):
                self.load_parameters(best_model_path)
                self.is_checkpoint_loaded = False
                self.is_best_model_loaded = True
            else:
                self.init_if_necessary()
        else:
            logger.info("The best model has been loaded")

    def _build_env(self):
        """
        building the program and strategy for specific running phase.
        """
        if self.env.is_inititalized:
            return

        self._build_env_start_event()
        self.env.is_inititalized = True
        self.env.main_program = clone_program(
            self._base_main_program, for_test=False)

        self.env.startup_program = fluid.Program()
        with fluid.program_guard(self.env.main_program,
                                 self._base_startup_program):
            with fluid.unique_name.guard(self.env.UNG):
                self.env.outputs = self._build_net()
                if self.is_train_phase or self.is_test_phase:
                    self.env.labels = self._add_label()
                    self.env.loss = self._add_loss()
                    self.env.metrics = self._add_metrics()

        if self.is_predict_phase or self.is_test_phase:
            self.env.main_program = clone_program(
                self.env.main_program, for_test=True)
            hub.common.paddle_helper.set_op_attr(
                self.env.main_program, is_test=True)

        if self.config.enable_memory_optim:
            for var_name in self.fetch_list:
                var = self.env.main_program.global_block().vars[var_name]
                var.persistable = True

        if self.is_train_phase:
            with fluid.program_guard(self.env.main_program,
                                     self._base_startup_program):
                with fluid.unique_name.guard(self.env.UNG):
                    self.scheduled_lr, self.max_train_steps = self.config.strategy.execute(
                        self.loss, self._base_data_reader, self.config,
                        self.device_count)

        if self.is_train_phase:
            loss_name = self.env.loss.name
        else:
            loss_name = None

        share_vars_from = self._base_compiled_program

        if not self.config.use_data_parallel:
            self.env.main_program_compiled = None
        else:
            self.env.main_program_compiled = fluid.CompiledProgram(
                self.env.main_program).with_data_parallel(
                    loss_name=loss_name,
                    share_vars_from=share_vars_from,
                    build_strategy=self.build_strategy,
                    places=self.places)

        self.exe.run(self.env.startup_program)
        self._build_env_end_event()

    @property
    def places(self):
        if self.config.use_cuda:
            _places = fluid.framework.cuda_places()
        else:
            _places = fluid.framework.cpu_places()

        if not self.config.use_data_parallel:
            return [_places[0]]
        return _places

    @property
    def return_numpy(self):
        return True

    @property
    def is_train_phase(self):
        return self.phase in ["train"]

    @property
    def is_test_phase(self):
        return self.phase in ["val", "dev", "test"]

    @property
    def is_predict_phase(self):
        return self.phase in ["predict", "inference"]

    @property
    def phase(self):
        return self._phases[-1]

    @property
    def env(self):
        phase = self.phase
        if phase in ["val", "dev", "test"]:
            phase = "dev"
        if not phase in self._envs:
            self._envs[phase] = RunEnv()
        return self._envs[phase]

    @property
    def py_reader(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.py_reader

    @property
    def current_step(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.current_step

    @property
    def current_epoch(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.current_epoch

    @property
    def main_program(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.main_program

    @property
    def startup_program(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.startup_program

    @property
    def main_program_compiled(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.main_program_compiled

    @property
    def main_program_to_be_run(self):
        if self.config.use_data_parallel:
            if self._base_compiled_program is None:
                self._base_compiled_program = self.env.main_program_compiled
            return self.main_program_compiled
        return self.main_program

    @property
    def reader(self):
        if self.is_predict_phase:
            data = self._predict_data
        else:
            data = None
        self.env.reader = self._base_data_reader.data_generator(
            batch_size=self.config.batch_size,
            phase=self.phase,
            data=data,
            return_list=not self.config.use_pyreader)
        return self.env.reader

    @property
    def loss(self):
        if self.is_predict_phase:
            raise RuntimeError()

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.loss

    @property
    def labels(self):
        if self.is_predict_phase:
            raise RuntimeError()

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.labels

    @property
    def outputs(self):
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.outputs

    @property
    def metrics(self):
        if self.is_predict_phase:
            raise RuntimeError()

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.metrics

    @property
    def unique_name_generator(self):
        return self.env.UNG

    @property
    def feed_list(self):
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase or self.is_test_phase:
            feed_list += [label.name for label in self.labels]
        return feed_list

    @property
    def feed_var_list(self):
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self.feed_list]

    @property
    def fetch_list(self):
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        return [output.name for output in self.outputs]

    @property
    def fetch_var_list(self):
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self.fetch_list]

    @property
    def tb_writer(self):
        """
        get tb_writer for visualization.
        """
        if not os.path.exists(self.config.checkpoint_dir):
            mkdir(self.config.checkpoint_dir)
        tb_log_dir = os.path.join(self.config.checkpoint_dir, "visualization")
        if not self._tb_writer:
            self._tb_writer = SummaryWriter(tb_log_dir)
        return self._tb_writer

    def create_event_function(self, hook_type):
        """
        create handlers for specific event.

        Args:
            hook_type (str): specific event name

        Returns:
            func: executable function, the class method will receive a parameter named self.
        """

        def hook_function(self, *args):
            # all the handler in self._hooks[hook_type] will be configured to executable
            for name, func in self._hooks[hook_type].items():
                if inspect.ismethod(func):
                    func(*args)
                else:
                    partial(func, self)(*args)

        return hook_function

    @property
    def hooks(self):
        return self._hooks

    def hooks_info(self, show_default=False):
        """
        get the hooks information, including the source code.

        Args:
            show_default (bool): show the information of Paddlehub default hooks or not, default False

        Returns:
            str: the formatted string of the hooks information
        """
        return self._hooks.info(show_default)

    def add_hook(self, hook_type, name=None, func=None):
        """
        add the handler function to spectific event.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name, default None
            func (func): the handler function, default None
        """
        if name == None:
            name = "hook_%s" % id(func)
        self._hooks.add(hook_type, name=name, func=func)
        logger.info("Add hook %s:%s successfully" % (hook_type, name))

    def delete_hook(self, hook_type, name):
        """
        delete the handler function of spectific event.

        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
        """
        self._hooks.delete(hook_type, name)
        logger.info("Delete hook %s:%s successfully" % (hook_type, name))

    def modify_hook(self, hook_type, name, func):
        """
         modify the handler function of spectific event.

         Args:
             hook_type (str): the spectific event name
             name (str): the handler function name
             func (func): the new handler function
         """
        self._hooks.modify(hook_type, name, func)
        logger.info("Modify hook %s:%s successfully" % (hook_type, name))

    def _default_build_env_start_event(self):
        pass

    def _default_build_env_end_event(self):
        if not self.is_predict_phase:
            self.env.score_scalar = {}

    def _default_finetune_start_event(self):
        logger.info("PaddleHub finetune start")

    def _default_finetune_end_event(self, run_states):
        logger.info("PaddleHub finetune finished.")

    def _default_predict_start_event(self):
        logger.info("PaddleHub predict start")

    def _default_predict_end_event(self, run_states):
        logger.info("PaddleHub predict finished.")

    def _default_eval_start_event(self):
        logger.info("Evaluation on {} dataset start".format(self.phase))

    def _default_eval_end_event(self, run_states):
        """
        Paddlehub default handler for eval_end_event, it will complete visualization and metrics calculation

        Args:
            run_states (object): the results in eval phase
        """
        eval_scores, eval_loss, run_speed = self._calculate_metrics(run_states)
        if 'train' in self._envs:
            self.tb_writer.add_scalar(
                tag="Loss_{}".format(self.phase),
                scalar_value=eval_loss,
                global_step=self._envs['train'].current_step)

        log_scores = ""
        for metric in eval_scores:
            if 'train' in self._envs:
                self.tb_writer.add_scalar(
                    tag="{}_{}".format(metric, self.phase),
                    scalar_value=eval_scores[metric],
                    global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, eval_scores[metric])
        logger.eval(
            "[%s dataset evaluation result] loss=%.5f %s[step/sec: %.2f]" %
            (self.phase, eval_loss, log_scores, run_speed))

        eval_scores_items = eval_scores.items()
        if len(eval_scores_items):
            # The first metric will be chose to eval
            main_metric, main_value = list(eval_scores_items)[0]
        else:
            logger.warning(
                "None of metrics has been implemented, loss will be used to evaluate."
            )
            # The larger, the better
            main_metric, main_value = "negative loss", -eval_loss
        if self.phase in ["dev", "val"] and main_value > self.best_score:
            self.best_score = main_value
            model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            logger.eval("best model saved to %s [best %s=%.5f]" %
                        (model_saved_dir, main_metric, main_value))
            self.save_inference_model(dirname=model_saved_dir)

    def _default_log_interval_event(self, run_states):
        """
        PaddleHub default handler for log_interval_event, it will complete visualization.

        Args:
            run_states (object): the results in train phase
        """
        scores, avg_loss, run_speed = self._calculate_metrics(run_states)
        self.tb_writer.add_scalar(
            tag="Loss_{}".format(self.phase),
            scalar_value=avg_loss,
            global_step=self._envs['train'].current_step)
        log_scores = ""
        for metric in scores:
            self.tb_writer.add_scalar(
                tag="{}_{}".format(metric, self.phase),
                scalar_value=scores[metric],
                global_step=self._envs['train'].current_step)
            log_scores += "%s=%.5f " % (metric, scores[metric])
        logger.train("step %d / %d: loss=%.5f %s[step/sec: %.2f]" %
                     (self.current_step, self.max_train_steps, avg_loss,
                      log_scores, run_speed))

    def _default_save_ckpt_interval_event(self):
        self.save_checkpoint()

    def _default_eval_interval_event(self):
        self.eval(phase="dev")

    def _default_run_step_event(self, run_state):
        pass

    def _build_net(self):
        raise NotImplementedError

    def _add_loss(self):
        raise NotImplementedError

    def _add_label(self):
        raise NotImplementedError

    def _add_metrics(self):
        # Some metrics like acc, auc can be calculated by fluid.layers
        # The others can be calculated in _calculate_metrics function
        raise NotImplementedError

    def _calculate_metrics(self, run_states):
        # NOTE: if you want to customize the metrics
        # you should make sure that the first parameter returned is a dict
        # The first key will be used as main metrics to update the best model
        raise NotImplementedError

    # NOTE: current saved checkpoint machanism is not completed,
    # it can't restore dataset training status
    def save_checkpoint(self):
        """
        save the program of the last step in training
        """
        model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                       "step_%d" % self.current_step)

        logger.info("Saving model checkpoint to {}".format(model_saved_dir))
        # to resume traning by loading ckpt, it must be save program (save_persistables)
        fluid.io.save_persistables(
            self.exe, dirname=model_saved_dir, main_program=self.main_program)
        save_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            current_epoch=self.current_epoch,
            global_step=self.current_step,
            best_score=self.best_score,
            exe=self.exe,
            main_program=self.main_program)

    def load_checkpoint(self):
        is_load_successful, self.env.current_epoch, self.env.current_step, self.best_score = load_checkpoint(
            self.config.checkpoint_dir,
            self.exe,
            main_program=self.main_program)

        return is_load_successful

    def load_parameters(self, dirname):
        def if_exist(var):
            path = os.path.join(dirname, var.name)
            return os.path.exists(path)

        fluid.io.load_vars(
            self.exe, dirname, self.main_program, predicate=if_exist)

    def save_parameters(self, dirname):
        fluid.io.save_params(
            self.exe, dirname=dirname, main_program=self.main_program)

    def save_inference_model(self,
                             dirname,
                             model_filename=None,
                             params_filename=None):
        with self.phase_guard("predict"):
            fluid.io.save_inference_model(
                dirname=dirname,
                executor=self.exe,
                feeded_var_names=self.feed_list,
                target_vars=self.fetch_var_list,
                main_program=self.main_program,
                model_filename=model_filename,
                params_filename=params_filename)

    def finetune_and_eval(self):
        return self.finetune(do_eval=True)

    def finetune(self, do_eval=False):
        """
        train and finetune the module parameters.

        Args:
            do_eval (bool): do eval during train phase or not

        Returns:
            RunState: the running result of train phase
        """

        # Start to finetune
        with self.phase_guard(phase="train"):
            self.init_if_necessary()
            self._finetune_start_event()
            run_states = []
            if self.current_epoch <= self.config.num_epoch:
                while self.current_epoch <= self.config.num_epoch:
                    self.config.strategy.step()
                    run_states = self._run(do_eval=do_eval)
                    self.env.current_epoch += 1

                # Final evaluation
                if self._base_data_reader.get_dev_examples() != []:
                    # Warning: DO NOT use self.eval(phase="dev", load_best_model=True) during training.
                    # It will cause trainer unable to continue training from checkpoint after eval.
                    # More important, The model should evaluate current performance during training.
                    self.eval(phase="dev")
                if self._base_data_reader.get_test_examples() != []:
                    self.eval(phase="test", load_best_model=True)
                # Save checkpoint after finetune
                self.save_checkpoint()

            self._finetune_end_event(run_states)
            return run_states

    def eval(self, phase="dev", load_best_model=False):
        """
        evaluate the performance of current module.

        Args:
            phase (str): current run phase
            load_best_model (bool): load the best model or not

        Returns:
            RunState: the running result of eval phase
        """
        # Warning: DO NOT use eval(load_best_model=True) in finetune_and_eval
        # It will cause trainer unable to continue training from checkpoint after eval
        # More important, The model should evaluate current performance during training.
        with self.phase_guard(phase=phase):
            if load_best_model:
                self.init_if_load_best_model()
            else:
                self.init_if_necessary()
            self._eval_start_event()
            run_states = self._run()
            self._eval_end_event(run_states)
            return run_states

    def _create_predictor(self):
        """
        create high-performance predictor for predict.

        Returns:
            PaddlePredictor: the high-performance predictor
        """
        with tmp_dir() as _dir:
            self.save_inference_model(dirname=_dir)
            predictor_config = fluid.core.AnalysisConfig(_dir)
            predictor_config.disable_glog_info()

            if self.config.use_cuda:
                predictor_config.enable_use_gpu(100, 0)
                predictor_config.switch_ir_optim(True)
            else:
                predictor_config.disable_gpu()
            predictor_config.enable_memory_optim()
            return fluid.core.create_paddle_predictor(predictor_config)

    def _run_with_predictor(self):
        """
        use high-performance predictor to make prediction.

        Returns:
            RunState: the running result of predict phase
        """

        if isinstance(self._base_data_reader, hub.reader.LACClassifyReader):
            raise Exception(
                "LACClassifyReader does not support predictor, please close accelerate_mode"
            )

        global_run_states = []
        period_run_states = []

        for run_step, batch in enumerate(self.reader(), start=1):
            step_run_state = RunState(len(self.fetch_list))
            step_run_state.run_step = 1
            num_batch_examples = len(batch)

            if not self.config.use_pyreader:
                # if use pyreader, the nlp_reader return [batch]
                batch = batch[0]

            batch = [fluid.core.PaddleTensor(data) for data in batch]
            fetch_result = self._predictor.run(batch)
            for index, result in enumerate(fetch_result):
                step_run_state.run_results[index] = result.as_ndarray()
            step_run_state.run_examples += num_batch_examples
            step_run_state.update()
            period_run_states += [step_run_state]
            self._run_step_event(step_run_state)

        global_run_states += period_run_states
        return global_run_states

    def predict(self,
                data,
                load_best_model=True,
                return_result=False,
                accelerate_mode=False):
        """
        make prediction for the input data.

        Args:
            data (list): the data will be predicted.
            load_best_model (bool): load the best model or not
            return_result (bool): return a readable result or just the raw run result
            accelerate_mode (bool): use high-performance predictor or not

        Returns:
            RunState: the running result of predict phase
        """
        self.accelerate_mode = accelerate_mode

        with self.phase_guard(phase="predict"):
            self._predict_data = data
            self._predict_start_event()

            if load_best_model:
                self.init_if_load_best_model()
            else:
                self.init_if_necessary()
            if not self.accelerate_mode:
                run_states = self._run()
            else:
                if not self._predictor:
                    self._predictor = self._create_predictor()
                run_states = self._run_with_predictor()

            self._predict_end_event(run_states)
            self._predict_data = None
            if return_result:
                return self._postprocessing(run_states)
        return run_states

    def _postprocessing(self, run_states):
        """
        postprocessing the run result, get readable result.

        Args:
            run_states (RunState): the raw run result to be processed

        Returns:
            list: readable result
        """
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results[0]
            results += [result[0] for result in batch_result]
        return results

    def _run(self, do_eval=False):
        """
        load data and run the program.

        Args:
            do_eval (bool): do eval during train phase or not

        Returns:
            RunState: the running result of specific phase
        """
        with fluid.program_guard(self.main_program, self.startup_program):
            if self.config.use_pyreader:
                data_loader = fluid.io.DataLoader.from_generator(
                    feed_list=self.feed_var_list,
                    capacity=64,
                    use_double_buffer=True,
                    iterable=True)
                data_reader = data_loader.set_batch_generator(
                    self.reader, places=self.places)
            else:
                data_feeder = fluid.DataFeeder(
                    feed_list=self.feed_list, place=self.place)
                data_reader = data_feeder.decorate_reader(
                    self.reader,
                    multi_devices=self.config.use_data_parallel,
                    drop_last=True)

            global_run_states = []
            period_run_states = []

            for run_step, batch in enumerate(data_reader(), start=1):
                step_run_state = RunState(len(self.fetch_list))
                step_run_state.run_step = 1
                num_batch_examples = len(batch)

                fetch_result = self.exe.run(
                    self.main_program_to_be_run,
                    feed=batch,
                    fetch_list=self.fetch_list,
                    return_numpy=self.return_numpy)
                if not self.return_numpy:
                    fetch_result = [np.array(x) for x in fetch_result]

                for index, result in enumerate(fetch_result):
                    step_run_state.run_results[index] = result
                step_run_state.run_examples += num_batch_examples
                step_run_state.update()
                period_run_states += [step_run_state]
                self.env.current_step += 1
                if self.is_train_phase:
                    if self.current_step % self.config.log_interval == 0:
                        self._log_interval_event(period_run_states)
                        global_run_states += period_run_states
                        period_run_states = []

                    if self.config.save_ckpt_interval and self.current_step % self.config.save_ckpt_interval == 0:
                        self._save_ckpt_interval_event()

                    if do_eval and self.current_step % self.config.eval_interval == 0:
                        self._eval_interval_event()

                self._run_step_event(step_run_state)

            global_run_states += period_run_states
            return global_run_states

    def __repr__(self):
        return "Task: %s with metrics_choices: %s， reader: %s, %s" % (
            self.__class__.__name__, self.metrics_choices,
            self._base_data_reader.__class__.__name__, self.config)
