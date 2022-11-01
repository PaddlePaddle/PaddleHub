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
import contextlib
import inspect
import os
from functools import partial
from typing import Any
from typing import Callable
from typing import Generator
from typing import Generic
from typing import Iterator
from typing import List
from typing import Union

import numpy as np
import paddle
from paddle.framework import core
from visualdl import LogWriter

from paddlehub.compat import paddle_utils
from paddlehub.compat.task.checkpoint import load_checkpoint
from paddlehub.compat.task.config import RunConfig
from paddlehub.compat.task.hook import TaskHooks
from paddlehub.compat.task.task_utils import RunEnv
from paddlehub.compat.task.task_utils import RunState
from paddlehub.utils.log import logger
from paddlehub.utils.utils import generate_tempdir


class BaseTask(object):
    '''
    BaseTask is the base class of all the task. It will complete the building of all the running environment.
    Args:
        main_program (object): the customized main_program, default None
        startup_program (object): the customized startup_program, default None
        config (object): the config for the task, default None
        metrics_choices (list): metrics used to the task, default ['acc']
    '''

    def __init__(self,
                 dataset: Iterator = None,
                 feed_list: List = None,
                 data_reader: Generic = None,
                 main_program: paddle.static.Program = None,
                 startup_program: paddle.static.Program = None,
                 config: RunConfig = None,
                 metrics_choices: List[str] = None):
        # metrics item
        self.best_score = -999
        if not metrics_choices:
            metrics_choices = ['acc']
        elif metrics_choices == None:
            metrics_choices = []
        if isinstance(metrics_choices, list):
            self.metrics_choices = metrics_choices
        else:
            self.metrics_choices = [metrics_choices]

        if main_program is None:
            self._base_main_program = paddle_utils.clone_program(paddle.static.default_main_program(), for_test=False)
        else:
            self._base_main_program = paddle_utils.clone_program(main_program, for_test=False)
        if startup_program is None:
            self._base_startup_program = paddle_utils.clone_program(paddle.static.default_startup_program(),
                                                                    for_test=False)
        else:
            self._base_startup_program = paddle_utils.clone_program(startup_program, for_test=False)
        self.is_checkpoint_loaded = False
        self._base_compiled_program = None

        # run config
        self.config = config if config else RunConfig()
        self.place = self.places[0]
        self.device_count = len(self.places)

        if self.config.use_data_parallel:
            if not self.config.use_pyreader and self.config.batch_size < self.device_count:
                logger.warning(
                    'Batch size({}) is less than the count of devices({}), which is not allowed in current Paddle versions'
                    .format(self.config.batch_size, self.device_count))
                logger.warning('Batch size automatically adjusted to {}'.format(self.device_count))
                self.config._batch_size = self.device_count

        self.exe = paddle.static.Executor(place=self.place)
        self.build_strategy = paddle.static.BuildStrategy()

        # run environment
        self._phases = []
        self._envs = {}
        self._predict_data = None
        self._vdl_writer = None

        # event hooks
        self._hooks = TaskHooks()
        for hook_type, event_hooks in self._hooks._registered_hooks.items():
            self._hooks.add(hook_type, 'default', eval('self._default_{}'.format(hook_type)))
            setattr(BaseTask, '_{}'.format(hook_type), self.create_event_function(hook_type))

        # accelerate predict
        self.is_best_model_loaded = False
        self._predictor = None

        # set default phase
        self.enter_phase('train')

        self.dataset = dataset
        if dataset:
            self._label_list = dataset.get_labels()
        else:
            self._label_list = None

        self._base_data_reader = data_reader
        self._base_feed_list = feed_list

        self._compatible_mode = True if data_reader else False

    @contextlib.contextmanager
    def phase_guard(self, phase: str):
        self.enter_phase(phase)
        yield
        self.exit_phase()

    def enter_phase(self, phase: str):
        if phase not in ['train', 'val', 'dev', 'test', 'predict', 'inference']:
            raise RuntimeError('Unknown phase {}.'.format(phase))
        if phase in ['val', 'dev']:
            phase = 'dev'
        elif phase in ['predict', 'inference']:
            phase = 'predict'
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
            best_model_path = os.path.join(self.config.checkpoint_dir, "best_model")
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
        '''Building the program and strategy for specific running phase.'''
        if self.env.is_inititalized:
            return

        self._build_env_start_event()
        self.env.is_inititalized = True
        self.env.main_program = paddle_utils.clone_program(self._base_main_program, for_test=False)

        self.env.startup_program = paddle.static.Program()
        with paddle.static.program_guard(self.env.main_program, self._base_startup_program):
            with paddle.utils.unique_name.guard(self.env.UNG):
                self.env.outputs = self._build_net()
                if self.is_train_phase or self.is_test_phase:
                    self.env.labels = self._add_label()
                    self.env.loss = self._add_loss()
                    self.env.metrics = self._add_metrics()

        if self.is_predict_phase or self.is_test_phase:
            self.env.main_program = paddle_utils.clone_program(self.env.main_program, for_test=True)
            paddle_utils.set_op_attr(self.env.main_program, is_test=True)

        if self.is_train_phase:
            with paddle.static.program_guard(self.env.main_program, self._base_startup_program):
                with paddle.utils.unique_name.guard(self.env.UNG):
                    if self._compatible_mode:
                        # This branch is compatible code for usage deprecated in paddlehub v1.8.
                        self._base_data_reader.data_generator(batch_size=self.config.batch_size,
                                                              phase='train',
                                                              shuffle=True)
                        num_train_examples = self._base_data_reader.num_examples['train']
                        try:
                            # nlp_reader
                            _in_tokens = self._base_data_reader.in_tokens
                            if _in_tokens:
                                num_train_examples *= self._base_data_reader.max_seq_len
                        except:
                            # cv_reader without .in_tokens and .max_seq_len
                            ...
                    else:
                        num_train_examples = len(self.dataset.get_train_records())

                    self.max_train_steps = self.config.num_epoch * num_train_examples // self.config.batch_size // self.device_count
                    self.scheduled_lr = self.config.strategy.execute(self.loss, self.max_train_steps)

        if self.is_train_phase:
            loss_name = self.env.loss.name
        else:
            loss_name = None

        share_vars_from = self._base_compiled_program

        if not self.config.use_data_parallel:
            self.env.main_program_compiled = None
        else:
            self.env.main_program_compiled = paddle.static.CompiledProgram(self.env.main_program).with_data_parallel(
                loss_name=loss_name,
                share_vars_from=share_vars_from,
                build_strategy=self.build_strategy,
                places=self.places)

        self.exe.run(self.env.startup_program)
        self._build_env_end_event()

    @property
    def places(self) -> List[Union[paddle.CPUPlace, paddle.CUDAPlace]]:
        if self.config.use_cuda:
            _places = paddle.device.framework.cuda_places()
        else:
            _places = paddle.device.framework.cpu_places()

        if not self.config.use_data_parallel:
            return [_places[0]]
        return _places

    @property
    def return_numpy(self) -> bool:
        return True

    @property
    def is_train_phase(self) -> bool:
        return self.phase in ['train']

    @property
    def is_test_phase(self) -> bool:
        return self.phase in ['val', 'dev', 'test']

    @property
    def is_predict_phase(self) -> bool:
        return self.phase in ['predict', 'inference']

    @property
    def phase(self) -> str:
        return self._phases[-1]

    @property
    def env(self) -> RunEnv:
        phase = self.phase
        if phase in ['val', 'dev', 'test']:
            phase = 'dev'
        if not phase in self._envs:
            self._envs[phase] = RunEnv()
        return self._envs[phase]

    @property
    def current_step(self) -> int:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.current_step

    @property
    def current_epoch(self) -> int:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.current_epoch

    @property
    def main_program(self) -> paddle.static.Program:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.main_program

    @property
    def startup_program(self) -> paddle.static.Program:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.startup_program

    @property
    def main_program_compiled(self) -> paddle.static.CompiledProgram:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.main_program_compiled

    @property
    def main_program_to_be_run(self) -> Union[paddle.static.Program, paddle.static.CompiledProgram]:
        if self.config.use_data_parallel:
            if self._base_compiled_program is None:
                self._base_compiled_program = self.env.main_program_compiled
            return self.main_program_compiled
        return self.main_program

    @property
    def generator(self) -> Generator:

        def data_generator(records):

            def wrapper():
                for record in records:
                    values = []
                    for feed_name in self.feed_list:
                        values.append(record[feed_name])
                    yield values

            return wrapper

        if self._compatible_mode:
            self.env.generator = self._base_data_reader.data_generator(batch_size=self.config.batch_size,
                                                                       phase=self.phase,
                                                                       data=self._predict_data,
                                                                       return_list=True)
        else:
            if self.is_predict_phase:
                records = self._predict_data
            else:
                if self.is_train_phase:
                    shuffle = True
                else:
                    shuffle = False
                records = self.dataset.get_records(phase=self.phase, shuffle=shuffle)
            self.env.generator = data_generator(records)

        return self.env.generator

    @property
    def loss(self) -> paddle.static.Variable:
        if self.is_predict_phase:
            raise RuntimeError('Loss cannot be obtained in the prediction phase.')

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.loss

    @property
    def labels(self) -> List[paddle.static.Variable]:
        if self.is_predict_phase:
            raise RuntimeError('Labels cannot be obtained in the prediction phase.')

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.labels

    @property
    def outputs(self) -> List[paddle.static.Variable]:
        if not self.env.is_inititalized:
            self._build_env()
        return self.env.outputs

    @property
    def metrics(self) -> List[str]:
        if self.is_predict_phase:
            raise RuntimeError('Metrics cannot be obtained in the prediction phase.')

        if not self.env.is_inititalized:
            self._build_env()
        return self.env.metrics

    @property
    def unique_name_generator(self):
        return self.env.UNG

    @property
    def feed_list(self) -> List[str]:
        if not self.env.is_inititalized:
            self._build_env()

        if self._predict_data:
            feed_list = list(self._predict_data[0].keys())
        else:
            feed_list = self.dataset.get_feed_list(self.phase)

        feed_list = [feed_name for feed_name in feed_list if feed_name in self.main_program.global_block().vars]
        return feed_list

    @property
    def feed_var_list(self) -> List[paddle.static.Variable]:
        if not self.env.is_inititalized:
            self._build_env()

        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self.feed_list]

    @property
    def fetch_list(self) -> List[str]:
        if self.is_train_phase or self.is_test_phase:
            return [metric.name for metric in self.metrics] + [self.loss.name]
        return [output.name for output in self.outputs]

    @property
    def fetch_var_list(self) -> List[paddle.static.Variable]:
        vars = self.main_program.global_block().vars
        return [vars[varname] for varname in self.fetch_list]

    @property
    def vdl_writer(self) -> LogWriter:
        '''
        get vdl_writer for visualization.
        '''
        if not os.path.exists(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
        tb_log_dir = os.path.join(self.config.checkpoint_dir, 'visualization')
        if not self._vdl_writer:
            self._vdl_writer = LogWriter(tb_log_dir)
        return self._vdl_writer

    def create_event_function(self, hook_type: str) -> Callable:
        '''
        create handlers for specific event.
        Args:
            hook_type (str): specific event name
        Returns:
            func: executable function, the class method will receive a parameter named self.
        '''

        def hook_function(self, *args):
            # all the handler in self._hooks[hook_type] will be configured to executable
            for name, func in self._hooks[hook_type].items():
                if inspect.ismethod(func):
                    func(*args)
                else:
                    partial(func, self)(*args)

        return hook_function

    @property
    def hooks(self) -> List[dict]:
        return self._hooks

    def hooks_info(self, show_default: bool = False) -> str:
        '''
        get the hooks information, including the source code.
        Args:
            show_default (bool): show the information of Paddlehub default hooks or not, default False
        Returns:
            str: the formatted string of the hooks information
        '''
        return self._hooks.info(show_default)

    def add_hook(self, hook_type: str, name: str = None, func: Callable = None):
        '''
        add the handler function to spectific event.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name, default None
            func (func): the handler function, default None
        '''
        if name == None:
            name = 'hook_{}'.format(id(func))
        self._hooks.add(hook_type, name=name, func=func)
        logger.info('Add hook {}:{} successfully'.format(hook_type, name))

    def delete_hook(self, hook_type: str, name: str):
        '''
        delete the handler function of spectific event.
        Args:
            hook_type (str): the spectific event name
            name (str): the handler function name
        '''
        self._hooks.delete(hook_type, name)
        logger.info('Delete hook {}:{} successfully'.format(hook_type, name))

    def modify_hook(self, hook_type: str, name: str, func: Callable):
        '''
         modify the handler function of spectific event.
         Args:
             hook_type (str): the spectific event name
             name (str): the handler function name
             func (func): the new handler function
         '''
        self._hooks.modify(hook_type, name, func)
        logger.info('Modify hook {}:{} successfully'.format(hook_type, name))

    def _default_build_env_start_event(self):
        ...

    def _default_build_env_end_event(self):
        if not self.is_predict_phase:
            self.env.score_scalar = {}

    def _default_finetune_start_event(self):
        logger.info('PaddleHub finetune start')

    def _default_finetune_end_event(self, run_states: List[RunState]):
        logger.info('PaddleHub finetune finished.')

    def _default_predict_start_event(self):
        logger.info('PaddleHub predict start')

    def _default_predict_end_event(self, run_states: List[RunState]):
        logger.info('PaddleHub predict finished.')

    def _default_eval_start_event(self):
        logger.info('Evaluation on {} dataset start'.format(self.phase))

    def _default_eval_end_event(self, run_states: List[RunState]):
        '''
        Paddlehub default handler for eval_end_event, it will complete visualization and metrics calculation
        Args:
            run_states (object): the results in eval phase
        '''
        eval_scores, eval_loss, run_speed = self._calculate_metrics(run_states)
        if 'train' in self._envs:
            self.vdl_writer.add_scalar(tag='Loss_{}'.format(self.phase),
                                       value=eval_loss,
                                       step=self._envs['train'].current_step)

        log_scores = ''
        for metric in eval_scores:
            if 'train' in self._envs:
                self.vdl_writer.add_scalar(tag='{}_{}'.format(metric, self.phase),
                                           value=eval_scores[metric],
                                           step=self._envs['train'].current_step)

            log_scores += '{}={:.5f} '.format(metric, eval_scores[metric])
        logger.eval('[{} dataset evaluation result] loss={:.5f} {}[step/sec: {:.2f}]'.format(
            self.phase, eval_loss, log_scores, run_speed))

        eval_scores_items = eval_scores.items()
        if len(eval_scores_items):
            # The first metric will be chose to eval
            main_metric, main_value = list(eval_scores_items)[0]
        else:
            logger.warning('None of metrics has been implemented, loss will be used to evaluate.')
            # The larger, the better
            main_metric, main_value = 'negative loss', -eval_loss
        if self.phase in ['dev', 'val'] and main_value > self.best_score:
            self.best_score = main_value
            model_saved_dir = os.path.join(self.config.checkpoint_dir, 'best_model')
            logger.eval('best model saved to {} [best {}={:.5f}]'.format(model_saved_dir, main_metric, main_value))
            self.save_inference_model(dirname=model_saved_dir)

    def _default_log_interval_event(self, run_states: List[RunState]):
        '''
        PaddleHub default handler for log_interval_event, it will complete visualization.
        Args:
            run_states (object): the results in train phase
        '''
        scores, avg_loss, run_speed = self._calculate_metrics(run_states)
        self.vdl_writer.add_scalar(tag='Loss_{}'.format(self.phase),
                                   value=avg_loss,
                                   step=self._envs['train'].current_step)
        log_scores = ''
        for metric in scores:
            self.vdl_writer.add_scalar(tag='{}_{}'.format(metric, self.phase),
                                       value=scores[metric],
                                       step=self._envs['train'].current_step)
            log_scores += '{}={:.5f} '.format(metric, scores[metric])
        logger.train('step {} / {}: loss={:.5f} {}[step/sec: {:.2f}]'.format(self.current_step, self.max_train_steps,
                                                                             avg_loss, log_scores, run_speed))

    def _default_save_ckpt_interval_event(self):
        self.save_checkpoint()

    def _default_eval_interval_event(self):
        self.eval(phase='dev')

    def _default_run_step_event(self, run_state: List[RunState]):
        ...

    def _build_net(self):
        raise NotImplementedError

    def _add_loss(self):
        raise NotImplementedError

    def _add_label(self):
        raise NotImplementedError

    def _add_metrics(self):
        # Some metrics like acc, auc
        # The others can be calculated in _calculate_metrics function
        raise NotImplementedError

    def _calculate_metrics(self, run_states: List[RunState]):
        # NOTE: if you want to customize the metrics
        # you should make sure that the first parameter returned is a dict
        # The first key will be used as main metrics to update the best model
        raise NotImplementedError

    def load_checkpoint(self):
        is_load_successful, self.env.current_epoch, self.env.current_step, self.best_score = load_checkpoint(
            self.config.checkpoint_dir, self.exe, main_program=self.main_program)

        # Revise max_train_steps when incremental training
        if is_load_successful:
            self.max_train_steps = self.env.current_step + self.max_train_steps / self.config.num_epoch * (
                self.config.num_epoch - self.env.current_epoch + 1)
        return is_load_successful

    def load_parameters(self, dirname):

        def if_exist(var):
            path = os.path.join(dirname, var.name)
            return os.path.exists(path)

        paddle.static.load(executor=self.exe, model_path=dirname, program=self.main_program)

    def save_inference_model(self, dirname: str, model_filename: str = None, params_filename: str = None):
        with self.phase_guard('predict'):
            paddle.static.save_inference_model(dirname=dirname,
                                               executor=self.exe,
                                               main_program=self.main_program,
                                               feeded_var_names=self.feed_list,
                                               target_vars=self.fetch_var_list,
                                               model_filename=model_filename,
                                               params_filename=params_filename)

    def finetune_and_eval(self) -> List[RunState]:
        return self.finetune(do_eval=True)

    def finetune(self, do_eval: bool = False) -> List[RunState]:
        '''
        train and finetune the module parameters.
        Args:
            do_eval (bool): do eval during train phase or not
        Returns:
            RunState: the running result of train phase
        '''

        # Start to finetune
        with self.phase_guard(phase='train'):
            self.init_if_necessary()
            self._finetune_start_event()
            run_states = []
            if self.current_epoch <= self.config.num_epoch:
                while self.current_epoch <= self.config.num_epoch:
                    self.config.strategy.step()
                    run_states = self._run(do_eval=do_eval)
                    self.env.current_epoch += 1

                # Final evaluation
                if self._compatible_mode:
                    dev_examples = self._base_data_reader.get_dev_examples()
                    test_examples = self._base_data_reader.get_test_examples()
                else:
                    dev_examples = self.dataset.get_dev_examples()
                    test_examples = self.dataset.get_test_examples()
                if dev_examples != []:
                    # Warning: DO NOT use self.eval(phase='dev', load_best_model=True) during training.
                    # It will cause trainer unable to continue training from checkpoint after eval.
                    # More important, The model should evaluate current performance during training.
                    self.eval(phase='dev')
                if test_examples != []:
                    self.eval(phase='test', load_best_model=True)

                # Save checkpoint after finetune
                self.save_checkpoint()

            self._finetune_end_event(run_states)
            return run_states

    def eval(self, phase: str = 'dev', load_best_model: bool = False) -> List[RunState]:
        '''
        evaluate the performance of current module.
        Args:
            phase (str): current run phase
            load_best_model (bool): load the best model or not
        Returns:
            RunState: the running result of eval phase
        '''
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

    def _create_predictor(self) -> core.PaddlePredictor:
        '''
        create high-performance predictor for predict.
        Returns:
            PaddlePredictor: the high-performance predictor
        '''
        with generate_tempdir() as _dir:
            self.save_inference_model(dirname=_dir)
            predictor_config = core.AnalysisConfig(_dir)
            predictor_config.disable_glog_info()

            if self.config.use_cuda:
                predictor_config.enable_use_gpu(100, 0)
                predictor_config.switch_ir_optim(True)
            else:
                predictor_config.disable_gpu()
            predictor_config.enable_memory_optim()
            return core.create_paddle_predictor(predictor_config)

    def _run_with_predictor(self) -> List[RunState]:
        '''
        use high-performance predictor to make prediction.
        Returns:
            RunState: the running result of predict phase
        '''
        global_run_states = []
        period_run_states = []

        feed_var_shape = []
        feed_var_type = []
        for var in self.feed_var_list:
            feed_var_shape.append(var.shape)
            feed_var_type.append(paddle_utils.dtype_map[var.dtype])

        data_reader = self.generator
        for batch in data_reader():

            step_run_state = RunState(len(self.fetch_list))
            step_run_state.run_step = 1
            num_batch_examples = len(batch)

            # Preocessing data to the suitable shape and type for the model
            processed_batch = [[] for i in range(len(self.feed_list))]

            for sample in batch:
                for i, data in enumerate(sample):
                    processed_batch[i].append(data)
            tensor_batch = [[] for i in range(len(self.feed_list))]
            for i in range(len(processed_batch)):
                processed_batch[i] = np.array(processed_batch[i]).reshape(feed_var_shape[i]).astype(feed_var_type[i])
                tensor_batch[i] = core.PaddleTensor(processed_batch[i])

            fetch_result = self._predictor.run(tensor_batch)
            for index, result in enumerate(fetch_result):
                step_run_state.run_results[index] = result.as_ndarray()
            step_run_state.run_examples += num_batch_examples
            step_run_state.update()
            period_run_states += [step_run_state]
            self._run_step_event(step_run_state)

        global_run_states += period_run_states
        return global_run_states

    def predict(
        self,
        data: List[Any] = None,
        label_list: List[Any] = None,
        load_best_model: bool = True,
        return_result: bool = True,
        accelerate_mode: bool = True,
    ) -> List[RunState]:
        '''
        make prediction for the input data.
        Args:
            data (list): the data will be predicted. Its element should be a record when the task is initialized without data_reader param,
                         or a plaintext string list when the task is initialized with data_reader param (deprecated in paddlehub v1.8).
            label_list (list): the label list, used to proprocess the output.
            return_result (bool): return a readable result or just the raw run result. Always True when the task is not initialized with data_reader but dataset parameter.
            accelerate_mode (bool): use high-performance predictor or not
        Returns:
            RunState: the running result of predict phase
        '''
        self.accelerate_mode = accelerate_mode

        with self.phase_guard(phase='predict'):
            self._predict_data = data
            if label_list:
                self._label_list = label_list
            self._predict_start_event()

            if load_best_model:
                self.init_if_load_best_model()

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

    def _postprocessing(self, run_states: List[RunState]) -> List:
        '''
        postprocessing the run result, get readable result.
        Args:
            run_states (RunState): the raw run result to be processed
        Returns:
            list: readable result
        '''
        results = []
        for batch_state in run_states:
            batch_result = batch_state.run_results[0]
            results += [result[0] for result in batch_result]
        return results

    def _run(self, do_eval: bool = False) -> List[RunState]:
        '''
        load data and run the program.
        Args:
            do_eval (bool): do eval during train phase or not
        Returns:
            RunState: the running result of specific phase
        '''
        with paddle.static.program_guard(self.main_program, self.startup_program):
            data_loader = paddle.io.DataLoader.from_generator(feed_list=self.feed_var_list,
                                                              capacity=64,
                                                              use_double_buffer=True,
                                                              iterable=True)
            if self.is_predict_phase:
                data_reader = data_loader.set_sample_generator(self.generator,
                                                               places=self.places,
                                                               batch_size=self.config.batch_size,
                                                               drop_last=False)
            else:
                data_reader = data_loader.set_sample_generator(self.generator,
                                                               places=self.places,
                                                               batch_size=self.config.batch_size,
                                                               drop_last=True)

            global_run_states = []
            period_run_states = []
            for batch in data_reader():
                step_run_state = RunState(len(self.fetch_list))
                step_run_state.run_step = 1

                # get the batch_data_size
                tmp_name = list(batch[0].keys())[0]
                tmp = np.array(batch[0][tmp_name])
                num_batch_examples = tmp.shape[0]

                fetch_result = self.exe.run(self.main_program_to_be_run,
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

    def __repr__(self) -> str:
        return 'Task: {} with metrics_choices: {}, {}'.format(self.__class__.__name__, self.metrics_choices,
                                                              self.config)
