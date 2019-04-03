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

import unittest
import paddlehub as hub
import paddle.fluid as fluid
from paddlehub.paddle_helper import from_param_to_flexible_data, from_flexible_data_to_param
from paddlehub import module_desc_pb2
from paddlehub.logger import logger


class TestParamAttrSerializeAndDeSerialize(unittest.TestCase):
    def test_convert_l1_regularizer(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    regularizer=fluid.regularizer.L1Decay(
                        regularization_coeff=1)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.regularizer.__class__ == param_dict[
                'regularizer'].__class__, "regularzier type convert error!"
            assert fc_w.regularizer._regularization_coeff == param_dict[
                'regularizer']._regularization_coeff, "regularzier value convert error!"

    def test_convert_l2_regularizer(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    regularizer=fluid.regularizer.L2Decay(
                        regularization_coeff=1.5)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.regularizer.__class__ == param_dict[
                'regularizer'].__class__, "regularzier type convert error!"
            assert fc_w.regularizer._regularization_coeff == param_dict[
                'regularizer']._regularization_coeff, "regularzier value convert error!"

    def test_convert_error_clip_by_value(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    gradient_clip=fluid.clip.ErrorClipByValue(max=1)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.gradient_clip_attr.__class__ == param_dict[
                'gradient_clip_attr'].__class__, "clip type convert error!"
            assert fc_w.gradient_clip_attr.max == param_dict[
                'gradient_clip_attr'].max, "clip value convert error!"
            assert fc_w.gradient_clip_attr.min == param_dict[
                'gradient_clip_attr'].min, "clip value convert error!"

    def test_convert_gradient_clip_by_value(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    gradient_clip=fluid.clip.GradientClipByValue(max=1)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.gradient_clip_attr.__class__ == param_dict[
                'gradient_clip_attr'].__class__, "clip type convert error!"
            assert fc_w.gradient_clip_attr.max == param_dict[
                'gradient_clip_attr'].max, "clip value convert error!"
            assert fc_w.gradient_clip_attr.min == param_dict[
                'gradient_clip_attr'].min, "clip value convert error!"

    def test_convert_gradient_clip_by_normal(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    gradient_clip=fluid.clip.GradientClipByNorm(clip_norm=1)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.gradient_clip_attr.__class__ == param_dict[
                'gradient_clip_attr'].__class__, "clip type convert error!"
            assert fc_w.gradient_clip_attr.clip_norm == param_dict[
                'gradient_clip_attr'].clip_norm, "clip value convert error!"

    def test_convert_gradient_clip_by_global_normal(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(
                    name="fc_w",
                    gradient_clip=fluid.clip.GradientClipByGlobalNorm(
                        clip_norm=1)))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.gradient_clip_attr.__class__ == param_dict[
                'gradient_clip_attr'].__class__, "clip type convert error!"
            assert fc_w.gradient_clip_attr.clip_norm == param_dict[
                'gradient_clip_attr'].clip_norm, "clip value convert error!"
            assert fc_w.gradient_clip_attr.group_name == param_dict[
                'gradient_clip_attr'].group_name, "clip value convert error!"

    def test_convert_trainable(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(name="fc_w", trainable=False))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.trainable.__class__ == param_dict[
                'trainable'].__class__, "trainable type convert error!"
            assert fc_w.trainable == param_dict[
                'trainable'], "trainable value convert error!"

    def test_convert_do_model_average(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(name="fc_w", do_model_average=True))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.do_model_average.__class__ == param_dict[
                'do_model_average'].__class__, "do_model_average type convert error!"
            assert fc_w.do_model_average == param_dict[
                'do_model_average'], "do_model_average value convert error!"

    def test_convert_optimize_attr(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            input = fluid.layers.data(name="test", shape=[1], dtype="float32")
            fluid.layers.fc(
                input=input,
                size=10,
                param_attr=fluid.ParamAttr(name="fc_w", learning_rate=5))
            fc_w = [
                param for param in
                fluid.default_main_program().global_block().iter_parameters()
            ][0]
            flexible_data = module_desc_pb2.FlexibleData()
            from_param_to_flexible_data(fc_w, flexible_data)
            param_dict = from_flexible_data_to_param(flexible_data)
            assert fc_w.optimize_attr.__class__ == param_dict[
                'optimize_attr'].__class__, "optimize_attr type convert error!"
            assert fc_w.optimize_attr == param_dict[
                'optimize_attr'], "optimize_attr value convert error!"


if __name__ == "__main__":
    unittest.main()
