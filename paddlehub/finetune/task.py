#  Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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
import collections
import time
import multiprocessing

import numpy as np
import paddle.fluid as fluid
from visualdl import LogWriter

import paddlehub as hub
from paddlehub.common.utils import mkdir
from paddlehub.common.logger import logger
from paddlehub.finetune.checkpoint import load_checkpoint, save_checkpoint
from paddlehub.finetune.evaluate import chunk_eval, calculate_f1
from paddlehub.finetune.config import RunConfig

__all__ = [
    "ClassifierTask", "ImageClassifierTask", "TextClassifierTask",
    "SequenceLabelTask"
]


class RunState(object):
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


class BasicTask(object):
    def __init__(self,
                 feed_list,
                 data_reader,
                 main_program=None,
                 startup_program=None,
                 config=None):
        self.data_reader = data_reader
        self.main_program = main_program if main_program else fluid.default_main_program(
        )
        self.startup_program = startup_program if startup_program else fluid.default_startup_program(
        )
        self.config = config if config else RunConfig()
        self.place, self.device_count = hub.common.get_running_device_info(
            self.config)
        self.exe = fluid.Executor(place=self.place)
        self.feed_list = feed_list
        self.metrics = []
        self.is_inititalized = False
        self.current_step = 0
        self.current_epoch = 0

    def _init_start_event(self):
        pass

    def _init_end_event(self):
        pass

    def _eval_start_event(self, phase):
        logger.info("Evaluation on {} dataset start".format(phase))

    def _eval_end_event(self, phase, run_state):
        logger.info("[%s dataset evaluation result] [step/sec: %.2f]" %
                    (phase, run_state.run_speed))

    def _log_interval_event(self, run_state):
        logger.info("step %d: [step/sec: %.2f]" % (self.current_step,
                                                   run_state.run_speed))

    def _save_ckpt_interval_event(self):
        self.save_checkpoint(self.current_epoch, self.current_step)

    def _eval_interval_event(self):
        self.eval(phase="dev")

    def _run_step_event(self, phase, run_state):
        if phase == "predict":
            yield run_state.run_results

    def _finetune_start_event(self):
        logger.info("PaddleHub finetune start")

    def _finetune_end_event(self, run_state):
        logger.info("PaddleHub finetune finished.")

    def _build_net(self):
        raise NotImplementedError

    def _add_loss(self):
        raise NotImplementedError

    def _add_label(self):
        raise NotImplementedError

    def _add_metrics(self):
        raise NotImplementedError

    def _init_if_necessary(self, load_best_model=False):
        if not self.is_inititalized:
            self._init_start_event()
            with fluid.program_guard(self.main_program):
                self.output = self._build_net()
                self.inference_program = self.main_program.clone(for_test=True)
                self._add_label()
                self._add_loss()
                self._add_metrics()
                self.test_program = self.main_program.clone(for_test=True)
                self.config.strategy.execute(self.loss, self.data_reader,
                                             self.config)

            self.loss.persistable = True
            for metrics in self.metrics:
                metrics.persistable = True
            self.output.persistable = True

            self.build_strategy = fluid.BuildStrategy()
            if self.config.enable_memory_optim:
                self.build_strategy.memory_optimize = True
            else:
                self.build_strategy.memory_optimize = False

            self.main_program_compiled = fluid.CompiledProgram(
                self.main_program).with_data_parallel(
                    loss_name=self.loss.name,
                    build_strategy=self.build_strategy)
            self.inference_program_compiled = fluid.CompiledProgram(
                self.inference_program).with_data_parallel(
                    share_vars_from=self.main_program_compiled,
                    build_strategy=self.build_strategy)
            self.test_program_compiled = fluid.CompiledProgram(
                self.test_program).with_data_parallel(
                    share_vars_from=self.main_program_compiled,
                    build_strategy=self.build_strategy)

            self.load_checkpoint(load_best_model=load_best_model)

            if not os.path.exists(self.config.checkpoint_dir):
                mkdir(self.config.checkpoint_dir)
            vdl_log_dir = os.path.join(self.config.checkpoint_dir, "vdllog")
            self.log_writer = LogWriter(vdl_log_dir, sync_cycle=1)
            self.is_inititalized = True
            self._init_end_event()

    # NOTE: current saved checkpoint machanism is not completed,
    # it can't restore dataset training status
    def save_checkpoint(self, epoch, step):
        save_checkpoint(
            checkpoint_dir=self.config.checkpoint_dir,
            current_epoch=self.current_epoch,
            global_step=self.current_step,
            exe=self.exe,
            main_program=self.main_program)

    def load_checkpoint(self, load_best_model=False):
        self.current_epoch, self.current_step = load_checkpoint(
            self.config.checkpoint_dir,
            self.exe,
            main_program=self.main_program)

        if load_best_model:
            model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            if os.path.exists(model_saved_dir):
                fluid.io.load_persistables(
                    executor=self.exe,
                    dirname=model_saved_dir,
                    main_program=self.main_program)

    def get_feed_list(self, phase):
        if phase in ["train", "dev", "val", "test"]:
            return self.feed_list + [self.label.name]
        return self.feed_list

    def get_fetch_list(self, phase):
        metrics_name = [metric.name for metric in self.metrics]
        if phase in ["train", "dev", "val", "test"]:
            return metrics_name + [self.loss.name]
        return [self.output.name]

    def finetune_and_eval(self):
        self.finetune(do_eval=True)

    def finetune(self, do_eval=False):
        self._init_if_necessary()
        self._finetune_start_event()
        run_states = []
        if self.current_epoch <= self.config.num_epoch:
            # Start to finetune
            with fluid.program_guard(self.main_program):
                while self.current_epoch <= self.config.num_epoch:
                    train_reader = self.data_reader.data_generator(
                        batch_size=self.config.batch_size, phase='train')
                    run_states = self._run(
                        train_reader,
                        phase="train",
                        do_eval=do_eval,
                        program_compiled=self.main_program_compiled)
                    self.current_epoch += 1

            # Save checkpoint after finetune
            self.save_checkpoint(self.current_epoch + 1, self.current_step)

            # Final evaluation
            self.eval(phase="dev")
            self.eval(phase="test")

        self._finetune_end_event(run_states)

    def eval(self, phase="dev"):
        self._init_if_necessary()
        self._eval_start_event(phase)
        with fluid.program_guard(self.test_program):
            test_reader = self.data_reader.data_generator(
                batch_size=self.config.batch_size, phase=phase)
            run_states = self._run(
                test_reader, phase=phase, program_compiled=self.test_program)

        self._eval_end_event(phase, run_states)

    def _run(self, reader, phase, do_eval=False, program_compiled=None):
        if program_compiled is None:
            program_compiled = self.main_program_compiled
        feed_list = self.get_feed_list(phase=phase)
        data_feeder = fluid.DataFeeder(feed_list=feed_list, place=self.place)
        fetch_list = self.get_fetch_list(phase=phase)
        global_run_states = []
        period_run_states = []

        for run_step, batch in enumerate(reader(), start=1):
            step_run_state = RunState(len(fetch_list))
            step_run_state.run_step = 1
            num_batch_examples = len(batch)

            fetch_result = self.exe.run(
                program_compiled,
                feed=data_feeder.feed(batch),
                fetch_list=fetch_list)

            for index, result in enumerate(fetch_result):
                step_run_state.run_results[index] = result
            step_run_state.run_examples += num_batch_examples
            step_run_state.update()
            period_run_states += [step_run_state]
            if phase == "train":
                self.current_step += 1
                if self.current_step % self.config.log_interval == 0:
                    self._log_interval_event(period_run_states)
                    global_run_states += period_run_states
                    period_run_states = []

                if self.config.save_ckpt_interval and self.current_step % self.config.save_ckpt_interval == 0:
                    self._save_ckpt_interval_event()

                if do_eval and self.current_step % self.config.eval_interval == 0:
                    self._eval_interval_event()

            self._run_step_event(phase, step_run_state)

        global_run_states += period_run_states
        return global_run_states

    def predict(self, data, load_best_model=True):
        self._init_if_necessary(load_best_model=load_best_model)
        with fluid.program_guard(self.inference_program):
            inference_reader = self.data_reader.data_generator(
                batch_size=self.config.batch_size, phase='predict', data=data)
            for run_state in self._run(
                    inference_reader,
                    phase='predict',
                    program_compiled=self.inference_program):
                yield run_state.run_results


class ClassifierTask(BasicTask):
    def __init__(self,
                 data_reader,
                 feature,
                 num_classes,
                 feed_list,
                 startup_program=None,
                 config=None,
                 hidden_units=None):

        main_program = feature.block.program

        super(ClassifierTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config)

        self.feature = feature
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.best_accuracy = -1

    def _build_net(self):
        cls_feats = self.feature
        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        return logits

    def _add_label(self):
        self.label = fluid.layers.data(name="label", dtype="int64", shape=[1])

    def _add_loss(self):
        ce_loss = fluid.layers.cross_entropy(
            input=self.output, label=self.label)
        self.loss = fluid.layers.mean(x=ce_loss)

    def _add_metrics(self):
        self.accuracy = fluid.layers.accuracy(
            input=self.output, label=self.label)
        self.metrics.append(self.accuracy)

    def _init_end_event(self):
        with self.log_writer.mode("train") as logw:
            self.train_loss_scalar = logw.scalar(tag="Loss [train]")
            self.train_acc_scalar = logw.scalar(tag="Accuracy [train]")
        with self.log_writer.mode("evaluate") as logw:
            self.eval_loss_scalar = logw.scalar(tag="Loss [eval]")
            self.eval_acc_scalar = logw.scalar(tag="Accuracy [eval]")

    def _calculate_metrics(self, run_states):
        loss_sum = acc_sum = run_examples = 0
        run_step = run_time_used = 0
        for run_state in run_states:
            run_examples += run_state.run_examples
            run_step += run_state.run_step
            loss_sum += np.mean(
                run_state.run_results[-1]) * run_state.run_examples
            acc_sum += np.mean(
                run_state.run_results[0]) * run_state.run_examples

        run_time_used = time.time() - run_states[0].run_time_begin
        avg_loss = loss_sum / run_examples
        avg_acc = acc_sum / run_examples
        run_speed = run_step / run_time_used

        return avg_loss, avg_acc, run_speed

    def _log_interval_event(self, run_states):
        avg_loss, avg_acc, run_speed = self._calculate_metrics(run_states)
        self.train_loss_scalar.add_record(self.current_step, avg_loss)
        self.train_acc_scalar.add_record(self.current_step, avg_acc)
        logger.info("step %d: loss=%.5f acc=%.5f [step/sec: %.2f]" %
                    (self.current_step, avg_loss, avg_acc, run_speed))

    def _eval_end_event(self, phase, run_states):
        eval_loss, eval_acc, run_speed = self._calculate_metrics(run_states)
        logger.info(
            "[%s dataset evaluation result] loss=%.5f acc=%.5f [step/sec: %.2f]"
            % (phase, eval_loss, eval_acc, run_speed))
        if phase in ["dev", "val"] and eval_acc > self.best_accuracy:
            self.eval_loss_scalar.add_record(self.current_step, eval_loss)
            self.eval_acc_scalar.add_record(self.current_step, eval_acc)
            self.best_accuracy = eval_acc
            model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            logger.info("best model saved to %s [best accuracy=%.5f]" %
                        (model_saved_dir, self.best_accuracy))
            save_result = fluid.io.save_persistables(
                executor=self.exe,
                dirname=model_saved_dir,
                main_program=self.main_program)


ImageClassifierTask = ClassifierTask


class TextClassifierTask(ClassifierTask):
    def __init__(self,
                 data_reader,
                 feature,
                 num_classes,
                 feed_list,
                 startup_program=None,
                 config=None,
                 hidden_units=None):

        main_program = feature.block.program

        super(TextClassifierTask, self).__init__(
            data_reader=data_reader,
            feature=feature,
            num_classes=num_classes,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config,
            hidden_units=hidden_units)

    def _build_net(self):
        cls_feats = fluid.layers.dropout(
            x=self.feature,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")

        if self.hidden_units is not None:
            for n_hidden in self.hidden_units:
                cls_feats = fluid.layers.fc(
                    input=cls_feats, size=n_hidden, act="relu")

        logits = fluid.layers.fc(
            input=cls_feats,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        return logits


class SequenceLabelTask(BasicTask):
    def __init__(
            self,
            feature,
            max_seq_len,
            num_classes,
            data_reader,
            feed_list,
            startup_program=None,
            config=None,
    ):

        main_program = feature.block.program

        super(SequenceLabelTask, self).__init__(
            data_reader=data_reader,
            main_program=main_program,
            feed_list=feed_list,
            startup_program=startup_program,
            config=config)

        self.feature = feature
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.best_f1 = -1

    def _build_net(self):
        self.logits = fluid.layers.fc(
            input=self.feature,
            size=self.num_classes,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name="cls_seq_label_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_seq_label_out_b",
                initializer=fluid.initializer.Constant(0.)))

        logits = self.logits
        logits = fluid.layers.flatten(logits, axis=2)
        logits = fluid.layers.softmax(logits)
        self.num_labels = logits.shape[1]
        return logits

    def _add_label(self):
        self.label = fluid.layers.data(
            name="label", shape=[self.max_seq_len, 1], dtype='int64')

    def _add_loss(self):
        labels = fluid.layers.flatten(self.label, axis=2)
        ce_loss = fluid.layers.cross_entropy(input=self.output, label=labels)
        self.loss = fluid.layers.mean(x=ce_loss)

    def _add_metrics(self):
        self.ret_labels = fluid.layers.reshape(x=self.label, shape=[-1, 1])
        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(self.logits, axis=2), shape=[-1, 1])
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64')
        self.seq_len = fluid.layers.assign(self.seq_len)
        self.metrics += [self.ret_labels, self.ret_infers, self.seq_len]

    def _init_end_event(self):
        with self.log_writer.mode("train") as logw:
            self.train_loss_scalar = logw.scalar(tag="Loss [train]")
        with self.log_writer.mode("evaluate") as logw:
            self.eval_f1_scalar = logw.scalar(tag="F1 [eval]")
            self.eval_precision_scalar = logw.scalar(tag="Precision [eval]")
            self.eval_recall_scalar = logw.scalar(tag="Recall [eval]")

    def _calculate_metrics(self, run_states):
        total_infer = total_label = total_correct = loss_sum = 0
        run_step = run_time_used = run_examples = 0
        for run_state in run_states:
            loss_sum += np.mean(run_state.run_results[-1])
            np_labels = run_state.run_results[0]
            np_infers = run_state.run_results[1]
            np_lens = run_state.run_results[2]
            label_num, infer_num, correct_num = chunk_eval(
                np_labels, np_infers, np_lens, self.num_labels,
                self.device_count)
            total_infer += infer_num
            total_label += label_num
            total_correct += correct_num
            run_examples += run_state.run_examples
            run_step += run_state.run_step

        run_time_used = time.time() - run_states[0].run_time_begin
        run_speed = run_step / run_time_used
        avg_loss = loss_sum / run_examples
        precision, recall, f1 = calculate_f1(total_label, total_infer,
                                             total_correct)
        return precision, recall, f1, avg_loss, run_speed

    def _log_interval_event(self, run_states):
        precision, recall, f1, avg_loss, run_speed = self._calculate_metrics(
            run_states)
        self.train_loss_scalar.add_record(self.current_step, avg_loss)
        logger.info("step %d: loss=%.5f [step/sec: %.2f]" %
                    (self.current_step, avg_loss, run_speed))

    def _eval_end_event(self, phase, run_states):
        precision, recall, f1, avg_loss, run_speed = self._calculate_metrics(
            run_states)
        self.eval_f1_scalar.add_record(self.current_step, f1)
        self.eval_precision_scalar.add_record(self.current_step, precision)
        self.eval_recall_scalar.add_record(self.current_step, recall)
        logger.info("[%s dataset evaluation result] [step/sec: %.2f]" %
                    (phase, run_speed))
        logger.info(
            "[%s evaluation] F1-Score=%f, precision=%f, recall=%f [step/sec: %.2f]"
            % (phase, f1, precision, recall, run_speed))
        if f1 > self.best_f1:
            self.best_f1 = f1
            model_saved_dir = os.path.join(self.config.checkpoint_dir,
                                           "best_model")
            logger.info("best model saved to %s [best F1=%.5f]" %
                        (model_saved_dir, self.best_f1))
            fluid.io.save_persistables(self.exe, dirname=model_saved_dir)

    def get_feed_list(self, phase):
        if phase in ["train", "dev", "val", "test"]:
            return self.feed_list + [self.label.name] + [self.seq_len.name]
        return self.feed_list
