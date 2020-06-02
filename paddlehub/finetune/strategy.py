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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid

from paddlehub.common.logger import logger
from paddlehub.finetune.regularizer import L2SPDecayRegularizer
import paddle.fluid.layers.learning_rate_scheduler as lr_scheduler
from paddle.fluid.layers import control_flow


def get_pretrained_parameter(main_program, start_program):
    pretrained_parameters = []
    global_block = main_program.global_block()
    for op in global_block.ops[::-1]:
        for input_arg in op.input_arg_names:
            var = global_block.var(input_arg)
            if isinstance(
                    var, fluid.framework.Parameter
            ) and input_arg not in start_program.global_block().vars:
                pretrained_parameters.append(var)

    return pretrained_parameters


def get_parentOp_depth_max(parent_ops, op_depth_dict):
    max_depth = 1
    for parent_op in parent_ops:
        depth = op_depth_dict.get(parent_op, 1)
        if max_depth < depth:
            max_depth = depth
    return max_depth


def get_opDepth_min(ops, op_depth_dict):
    min_depth = max(op_depth_dict.values())
    for op in ops:
        depth = op_depth_dict[op]
        if min_depth > depth:
            min_depth = depth
    return min_depth


def get_depth_parameter(main_program):
    global_block = main_program.global_block()

    var_op_dict = {}
    for op in global_block.ops:

        for input_arg in op.input_arg_names:
            if input_arg not in var_op_dict.keys():
                var_op_dict[input_arg] = {"output_ops": [], "input_ops": []}
            var_op_dict[input_arg]["output_ops"].append(op)

        for output_arg in op.output_arg_names:
            if output_arg not in var_op_dict.keys():
                var_op_dict[output_arg] = {"output_ops": [], "input_ops": []}
            var_op_dict[output_arg]["input_ops"].append(op)

    op_depth_dict = {}
    for op in global_block.ops:
        parent_ops = []
        for input_arg in op.input_arg_names:
            for parent_op in var_op_dict[input_arg]["input_ops"]:
                if parent_op not in parent_ops:
                    parent_ops.append(parent_op)
        if not parent_ops:
            op_depth_dict[op] = 1
        else:
            op_depth_dict[op] = get_parentOp_depth_max(parent_ops,
                                                       op_depth_dict) + 1

    depth_params_dict = {}
    updated_depth_params_dict = {}
    for param in global_block.iter_parameters():
        adherent_ops = var_op_dict[param.name]["output_ops"]
        depth = get_opDepth_min(adherent_ops, op_depth_dict)
        if depth not in depth_params_dict.keys():
            depth_params_dict[depth] = []
            updated_depth_params_dict[depth] = []
        depth_params_dict[depth].append(param)
        updated_depth_params_dict[depth].append(param)

    depth_list = sorted(depth_params_dict.keys())
    len_depth_list = len(depth_list)
    for index, depth in enumerate(depth_list):
        for param in depth_params_dict[depth]:
            prefix = param.name.split(".")[0]
            if index < len_depth_list - 1:
                next_depth = depth_list[index + 1]
                for param_next_depth in depth_params_dict[next_depth]:
                    prefix_next_depth = param_next_depth.name.split(".")[0]
                    if prefix == prefix_next_depth:
                        updated_depth_params_dict[depth].append(
                            param_next_depth)
                        updated_depth_params_dict[next_depth].remove(
                            param_next_depth)

                        if not updated_depth_params_dict[next_depth]:
                            updated_depth_params_dict.pop(next_depth)

    return updated_depth_params_dict


def set_gradual_unfreeze(depth_params_dict, unfreeze_depths):
    for depth in unfreeze_depths:
        for index, param in enumerate(depth_params_dict[depth]):
            depth_params_dict[depth][index].stop_gradient = False

    freeze_depths = list(
        set(depth_params_dict.keys()).difference(set(unfreeze_depths)))
    for depth in freeze_depths:
        for index, param in enumerate(depth_params_dict[depth]):
            depth_params_dict[depth][index].stop_gradient = True


class DefaultStrategy(object):
    def __init__(self, learning_rate=1e-4, optimizer_name="adam", **kwargs):
        self.learning_rate = learning_rate
        self._optimizer_name = optimizer_name
        if self._optimizer_name.lower() == "sgd":
            self.optimizer = fluid.optimizer.SGD(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "adagrad":
            self.optimizer = fluid.optimizer.Adagrad(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "adamax":
            self.optimizer = fluid.optimizer.Adamax(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "ftrl":
            self.optimizer = fluid.optimizer.Ftrl(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "larsmomentum":
            self.optimizer = fluid.optimizer.LarsMomentum(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "momentum":
            self.optimizer = fluid.optimizer.Momentum(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "decayedadagrad":
            self.optimizer = fluid.optimizer.DecayedAdagrad(
                learning_rate=self.learning_rate, **kwargs)
        elif self._optimizer_name.lower() == "rmsprop":
            self.optimizer = fluid.optimizer.RMSPropOptimizer(
                learning_rate=self.learning_rate, **kwargs)
        else:
            self.optimizer = fluid.optimizer.Adam(
                learning_rate=self.learning_rate, **kwargs)

    def execute(self, loss, max_train_steps):
        if self.optimizer is not None:
            self.optimizer.minimize(loss)
        else:
            raise ValueError("DefaultStrategy's optimizer is None")

    def __str__(self):
        return "DefaultStrategy"

    def step(self):
        pass


class CombinedStrategy(DefaultStrategy):
    def __init__(self,
                 optimizer_name="adam",
                 learning_rate=1e-4,
                 scheduler=None,
                 regularization=None,
                 clip=None,
                 **kwargs):
        super(CombinedStrategy, self).__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            **kwargs)
        self.kwargs = kwargs
        # init set
        self.scheduler = {
            "warmup": 0.0,
            "linear_decay": {
                "start_point": 1.0,
                "end_learning_rate": 0.0,
            },
            "noam_decay": False,
            "discriminative": {
                "blocks": 0,
                "params_layer": None,
                "factor": 2.6
            },
            "gradual_unfreeze": {
                "blocks": 0,
                "params_layer": None,
            },
            "slanted_triangle": {
                "cut_fraction": 0.0,
                "ratio": 32
            }
        }

        self.regularization = {
            "L2": 0.0,
            "L2SP": 0.0,
            "weight_decay": 0.0,
        }

        self.clip = {"GlobalNorm": 0.0, "Norm": 0.0}

        if scheduler == None:
            scheduler = {}
        if regularization == None:
            regularization = {}
        if clip == None:
            clip = {}

        # check legality and assign
        for name in scheduler:
            self.check_assign(self.scheduler, name, scheduler[name])
        for name in regularization:
            self.check_assign(self.regularization, name, regularization[name])
        for name in clip:
            self.check_assign(self.clip, name, clip[name])

        # resolve the conflict
        if self.scheduler["discriminative"]["params_layer"] and self.scheduler[
                "discriminative"]["blocks"]:
            logger.warning(
                "Both params_layer and blocks have been set in discriminative, only params_layer will take effect"
            )
            self.scheduler["discriminative"]["blocks"] = 0

        if self.scheduler["gradual_unfreeze"][
                "params_layer"] and self.scheduler["gradual_unfreeze"]["blocks"]:
            logger.warning(
                "Both params_layer and blocks have been set in gradual_unfreeze, only params_layer will take effect"
            )
            self.scheduler["gradual_unfreeze"]["blocks"] = 0

        if self.scheduler["slanted_triangle"]["cut_fraction"] and (
                self.scheduler["warmup"] or self.scheduler["noam_decay"]
                or self.scheduler["linear_decay"]["start_point"] < 1):
            logger.warning(
                "You are using slanted_triangle learning rate strategy, "
                "which will make warmup, noam_decay and linear_decay useless")
            self.scheduler["warmup"] = 0.0
            self.scheduler["noam_decay"] = False
            self.scheduler["linear_decay"]["start_point"] = 1

        if self.scheduler["noam_decay"] and self.scheduler["linear_decay"][
                "start_point"]:
            logger.warning(
                "Both noam_decay and linear_decay have been set, only noam_decay will take effect"
            )
            self.scheduler["linear_decay"]["start_point"] = 1

        self.epoch = 0
        self.main_program = None

    def check_assign(self, dictionary, key, value):
        if key not in dictionary:
            raise ValueError("Invalid parameter: %s" % key)
        if isinstance(value, dict) and isinstance(dictionary[key], dict):
            sub_dict = dictionary[key]
            for sub_name in value:
                self.check_assign(sub_dict, sub_name, value[sub_name])
        elif isinstance(dictionary[key], type(value)) or (
                isinstance(dictionary[key], float)
                and isinstance(value, (float, int))) or dictionary[key] == None:
            dictionary[key] = value
        else:
            if isinstance(dictionary[key], dict):
                raise ValueError(
                    "The type of parameter %s should be a dict with keys: %s" %
                    (key, dictionary[key].keys()))
            else:
                raise ValueError("The type of parameter %s should be %s" %
                                 (key, type(dictionary[key])))

    def add_scheduler(self, name="warmup", value=0, **values):
        if values:
            self.check_assign(self.scheduler, name, values)
        else:
            self.check_assign(self.scheduler, name, value)

    def add_regularization(self, name="L2", value=1e-3, **values):
        if values:
            self.check_assign(self.regularization, name, values)
        else:
            self.check_assign(self.regularization, name, value)

    def add_clip(self, name="GlobalNorm", value=1.0, **values):
        if values:
            self.check_assign(self.clip, name, values)
        else:
            self.check_assign(self.clip, name, value)

    def scheduler_handler(self, max_train_steps):
        scheduled_lr = fluid.layers.create_global_var(
            shape=[1],
            value=self.learning_rate,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        warmup_steps = int(max_train_steps * self.scheduler["warmup"])

        # noam_decay (based on warmup)
        if self.scheduler["noam_decay"]:
            if warmup_steps > 0:
                scheduled_lr = fluid.layers.learning_rate_scheduler \
                    .noam_decay(1 / (warmup_steps * (self.learning_rate ** 2)),
                                warmup_steps)
            else:
                logger.warning(
                    "Noam decay learning rate scheduler should have positive \
                    warmup steps, using constant learning rate instead!")

        # warmup, linear_decay
        if warmup_steps > 0 or self.scheduler["linear_decay"]["start_point"] < 1:
            with self.main_program._lr_schedule_guard():
                global_step = lr_scheduler._decay_step_counter()
                with control_flow.Switch() as switch:
                    if warmup_steps > 0:
                        with switch.case(global_step < warmup_steps):
                            decayed_lr = self.learning_rate * global_step * 1.0 / warmup_steps
                            fluid.layers.assign(decayed_lr, scheduled_lr)
                    if self.scheduler["linear_decay"]["start_point"] < 1:
                        linear_decay_start = int(
                            max_train_steps *
                            self.scheduler["linear_decay"]["start_point"])
                        if linear_decay_start < warmup_steps:
                            logger.warning(
                                "linear decay can not start during warmup process,"
                                "it will start after warmup ends!")
                            linear_decay_start = warmup_steps
                        with switch.case(global_step >= linear_decay_start):
                            decayed_lr = lr_scheduler.polynomial_decay(
                                learning_rate=self.learning_rate,
                                decay_steps=max_train_steps,
                                end_learning_rate=self.scheduler["linear_decay"]
                                ["end_learning_rate"],
                                power=1.0,
                                cycle=False)
                            fluid.layers.assign(decayed_lr, scheduled_lr)

        # slanted_triangle
        if self.scheduler["slanted_triangle"]["cut_fraction"]:
            cut_step = int(max_train_steps *
                           self.scheduler["slanted_triangle"]["cut_fraction"])
            ratio = self.scheduler["slanted_triangle"]["ratio"]
            global_step = lr_scheduler._decay_step_counter()
            with control_flow.Switch() as switch:
                with switch.case(global_step <= cut_step):
                    pct = global_step / cut_step
                    decayed_lr = self.learning_rate * (1 + pct *
                                                       (ratio - 1)) / ratio
                    fluid.layers.assign(decayed_lr, scheduled_lr)
                with switch.default():
                    pct = 1 - (global_step - cut_step) / (
                        max_train_steps - cut_step)
                    decayed_lr = self.learning_rate * (1 + pct *
                                                       (ratio - 1)) / ratio
                    fluid.layers.assign(decayed_lr, scheduled_lr)

        # set optimizer
        super(CombinedStrategy, self).__init__(
            optimizer_name=self._optimizer_name,
            learning_rate=scheduled_lr,
            **self.kwargs)

        # discriminative learning rate
        # based on layer
        if self.scheduler["discriminative"]["params_layer"]:
            max_layer = max(
                self.scheduler["discriminative"]["params_layer"].values())
            for param in self.main_program.global_block().iter_parameters():
                if param.name in self.scheduler["discriminative"][
                        "params_layer"]:
                    param_layer = self.scheduler["discriminative"][
                        "params_layer"][param.name]
                    param.optimize_attr["learning_rate"] *= pow(
                        1.0 / self.scheduler["discriminative"]["factor"],
                        max_layer - param_layer)

        # based on blocks
        if self.scheduler["discriminative"]["blocks"]:
            _block_layers = math.ceil(
                len(self.sorted_depth) /
                self.scheduler["discriminative"]["blocks"])
            power = 0
            for cnt, depth in enumerate(self.sorted_depth):
                for index, param in enumerate(self.depth_params_dict[depth]):
                    param.optimize_attr["learning_rate"] *= \
                        pow(1.0 / self.scheduler["discriminative"]["factor"], power)
                if cnt and cnt % _block_layers == 0:
                    power += 1
        return scheduled_lr

    def clip_handler(self):
        if self.clip["GlobalNorm"]:
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=self.clip["GlobalNorm"]))
        elif self.clip["Norm"]:
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByNorm(clip_norm=self.clip["Norm"]))

    def regularization_handler(self, loss, scheduled_lr):
        if self.regularization["L2"]:
            for param in self.main_program.global_block().all_parameters():
                param.regularizer = fluid.regularizer.L2Decay(
                    regularization_coeff=self.regularization["L2"])

        pretrained_params = get_pretrained_parameter(
            self.main_program, fluid.default_startup_program())

        if self.regularization["L2SP"]:
            #TODO: L2SP can only run in one process now
            for index, param in enumerate(pretrained_params):
                param.regularizer = L2SPDecayRegularizer(
                    regularization_coeff=self.regularization["L2SP"])

        _, param_grads = self.optimizer.minimize(loss)

        if self.regularization["weight_decay"]:
            param_list = {}
            for param in self.main_program.global_block().all_parameters():
                param_list[param.name] = param * 1.0
                param_list[param.name].stop_gradient = True

            for param, grad in param_grads:
                if self.exclude_from_weight_decay(param.name):
                    continue
                with param.block.program._optimized_guard(
                    [param, grad]), fluid.framework.name_scope("weight_decay"):
                    updated_param = param - param_list[
                        param.name] * self.regularization[
                            "weight_decay"] * scheduled_lr
                    fluid.layers.assign(output=param, input=updated_param)

    def execute(self, loss, max_train_steps):
        # base information
        self.main_program = loss.block.program

        if self.scheduler["discriminative"]["blocks"] > 0 or self.scheduler[
                "gradual_unfreeze"]["blocks"] > 0:
            self.depth_params_dict = get_depth_parameter(self.main_program)
            self.sorted_depth = sorted(
                self.depth_params_dict.keys(), reverse=True)
            self.max_depth = len(self.sorted_depth)

        # handle scheduler
        scheduled_lr = self.scheduler_handler(max_train_steps)

        # handle clip
        self.clip_handler()

        # handle regularization
        self.regularization_handler(loss, scheduled_lr)

        logger.info(self.__str__())
        return scheduled_lr

    def exclude_from_weight_decay(self, name):
        if name.find("layer_norm") > -1:
            return True
        bias_suffix = ["_bias", "_b", ".b_0"]
        for suffix in bias_suffix:
            if name.endswith(suffix):
                return True
        return False

    def step(self):
        if self.scheduler["gradual_unfreeze"]["blocks"] > 0:
            self.epoch += 1
            if self.max_depth > 0 and self.epoch <= self.scheduler[
                    "gradual_unfreeze"]["blocks"]:
                set_gradual_unfreeze(
                    depth_params_dict=self.depth_params_dict,
                    unfreeze_depths=self.
                    sorted_depth[:self.max_depth * self.epoch //
                                 self.scheduler["gradual_unfreeze"]["blocks"]])
        elif self.scheduler["gradual_unfreeze"]["params_layer"]:
            max_layer = max(
                self.scheduler["gradual_unfreeze"]["params_layer"].values())
            if self.epoch <= max_layer:
                for param in self.main_program.global_block().iter_parameters():
                    if param.name in self.scheduler["gradual_unfreeze"][
                            "params_layer"]:
                        param_layer = self.scheduler["gradual_unfreeze"][
                            "params_layer"][param.name]
                        if param_layer >= max_layer - self.epoch:
                            param.stop_gradient = False
                        else:
                            param.stop_gradient = True
            self.epoch += 1
        else:
            pass

    def __str__(self):
        self.clip = {"GlobalNorm": 0.0, "Norm": 0.0}

        strategy_name = ""
        strategy_name += "warmup, " if self.scheduler["warmup"] else ""
        strategy_name += "linear decay, " if self.scheduler["linear_decay"][
            "start_point"] < 1 else ""
        strategy_name += "noam decay, " if self.scheduler["noam_decay"] else ""
        strategy_name += "discriminative learning rate, " if self.scheduler[
            "discriminative"]["blocks"] or self.scheduler["discriminative"][
                "params_layer"] else ""
        strategy_name += "gradual unfreeze, " if self.scheduler[
            "gradual_unfreeze"]["blocks"] or self.scheduler["gradual_unfreeze"][
                "params_layer"] else ""
        strategy_name += "slanted triangle learning rate, " if self.scheduler[
            "slanted_triangle"] else ""

        strategy_name += "L2 regularization, " if self.regularization[
            "L2"] else ""
        strategy_name += "L2SP regularization, " if self.regularization[
            "L2SP"] else ""
        strategy_name += "weight decay regularization, " if self.regularization[
            "weight_decay"] else ""

        strategy_name += "GlobalNorm clip, " if self.clip["GlobalNorm"] else ""
        strategy_name += "Norm clip, " if self.clip["Norm"] else ""

        return "Strategy with %s" % (strategy_name)


class AdamWeightDecayStrategy(CombinedStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 lr_scheduler="linear_decay",
                 warmup_proportion=0.1,
                 weight_decay=0.01,
                 optimizer_name="adam",
                 **kwargs):
        scheduler = {"warmup": warmup_proportion}
        if lr_scheduler == "noam_decay":
            scheduler["noam_decay"] = True
        elif lr_scheduler == "linear_decay":
            scheduler["linear_decay"] = {
                "start_point": warmup_proportion,
                "end_learning_rate": 0,
            }
        else:
            raise ValueError("lr_scheduler {} is not setup "
                             "correctly".format(lr_scheduler))
        regularization = {"weight_decay": weight_decay}
        clip = {"GlobalNorm": 1.0}
        super(AdamWeightDecayStrategy, self).__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            scheduler=scheduler,
            regularization=regularization,
            clip=clip,
            **kwargs)


class L2SPFinetuneStrategy(CombinedStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3,
                 **kwargs):
        scheduler = {}
        regularization = {"L2SP": regularization_coeff}
        clip = {}
        super(L2SPFinetuneStrategy, self).__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            scheduler=scheduler,
            regularization=regularization,
            clip=clip,
            **kwargs)


class DefaultFinetuneStrategy(CombinedStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 regularization_coeff=1e-3,
                 **kwargs):
        scheduler = {}
        regularization = {"L2": regularization_coeff}
        clip = {}

        super(DefaultFinetuneStrategy, self).__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            scheduler=scheduler,
            regularization=regularization,
            clip=clip,
            **kwargs)


class ULMFiTStrategy(CombinedStrategy):
    def __init__(self,
                 learning_rate=1e-4,
                 optimizer_name="adam",
                 cut_fraction=0.1,
                 ratio=32,
                 dis_blocks=3,
                 factor=2.6,
                 dis_params_layer=None,
                 frz_blocks=3,
                 frz_params_layer=None,
                 **kwargs):

        scheduler = {
            "slanted_triangle": {
                "cut_fraction": cut_fraction,
                "ratio": ratio
            },
            "gradual_unfreeze": {
                "blocks": frz_blocks,
                "params_layer": frz_params_layer
            },
            "discriminative": {
                "blocks": dis_blocks,
                "factor": factor,
                "params_layer": dis_params_layer
            }
        }
        regularization = {}
        clip = {}
        super(ULMFiTStrategy, self).__init__(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            scheduler=scheduler,
            regularization=regularization,
            clip=clip,
            **kwargs)
