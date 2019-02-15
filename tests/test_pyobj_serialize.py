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

import sys
import math
import unittest
import paddle_hub as hub
import paddle.fluid as fluid
from paddle_hub.utils import from_pyobj_to_flexible_data, from_flexible_data_to_pyobj, get_pykey
from paddle_hub import module_desc_pb2
from paddle_hub.logger import logger


def _compare_float(a, b):
    error_value = 1.0e-9
    # if python version < 3.5
    if sys.version_info < (3, 5):
        return abs(a - b) < error_value
    else:
        return math.isclose(a, b)


def _check_none(pyobj, flexible_data):
    assert flexible_data.type == module_desc_pb2.NONE, "type conversion error"


def _check_int(pyobj, flexible_data):
    assert flexible_data.type == module_desc_pb2.INT, "type conversion error"
    assert flexible_data.i == pyobj, "value convesion error"


def _check_float(pyobj, flexible_data):
    assert flexible_data.type == module_desc_pb2.FLOAT, "type conversion error"
    assert _compare_float(flexible_data.f, pyobj), "value convesion error"


def _check_str(pyobj, flexible_data):
    assert flexible_data.type == module_desc_pb2.STRING, "type conversion error"
    assert flexible_data.s == pyobj, "value convesion error"


def _check_bool(pyobj, flexible_data):
    assert flexible_data.type == module_desc_pb2.BOOLEAN, "type conversion error"
    assert flexible_data.b == pyobj, "value convesion error"


class TestPyobj2FlexibleData(unittest.TestCase):
    def test_int_2_flexible_data(self):
        input = None
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        _check_none(input, flexible_data)

    def test_int_2_flexible_data(self):
        input = 1
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        _check_int(input, flexible_data)

    def test_float_2_flexible_data(self):
        input = 2.012
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        _check_float(input, flexible_data)

    def test_string_2_flexible_data(self):
        input = "123"
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        _check_str(input, flexible_data)

    def test_bool_2_flexible_data(self):
        input = False
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        _check_bool(input, flexible_data)

    def test_list_2_flexible_data(self):
        input = [1, 2, 3]
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        assert flexible_data.type == module_desc_pb2.LIST, "type conversion error"
        assert len(
            flexible_data.list.data) == len(input), "value convesion error"
        for index in range(len(input)):
            _check_int(input[index], flexible_data.list.data[str(index)])

    def test_tuple_2_flexible_data(self):
        input = (1, 2, 3)
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        assert flexible_data.type == module_desc_pb2.LIST, "type conversion error"
        assert len(
            flexible_data.list.data) == len(input), "value convesion error"
        for index in range(len(input)):
            _check_int(input[index], flexible_data.list.data[str(index)])

    def test_set_2_flexible_data(self):
        input = {1, 2, 3}
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        assert flexible_data.type == module_desc_pb2.SET, "type conversion error"
        assert len(
            flexible_data.set.data) == len(input), "value convesion error"
        for index in range(len(input)):
            assert flexible_data.set.data[str(
                index)].i in input, "value convesion error"

    def test_dict_2_flexible_data(self):
        input = {1: 1, 2: 2, 3: 3}
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        assert flexible_data.type == module_desc_pb2.MAP, "type conversion error"
        assert len(
            flexible_data.map.data) == len(input), "value convesion error"
        for key, value in flexible_data.map.data.items():
            realkey = get_pykey(key, flexible_data.map.keyType[key])
            assert realkey in input, "key convesion error"
            _check_int(input[realkey], flexible_data.map.data[key])

    def test_obj_2_flexible_data(self):
        class TestObj:
            def __init__(self):
                self.a = 1
                self.b = 2.0
                self.c = "str"
                self.d = {'a': 123}

        input = TestObj()
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        assert flexible_data.type == module_desc_pb2.OBJECT, "type conversion error"
        assert len(flexible_data.object.data) == len(
            input.__dict__), "value convesion error"
        _check_int(input.a, flexible_data.object.data['a'])
        _check_float(input.b, flexible_data.object.data['b'])
        _check_str(input.c, flexible_data.object.data['c'])
        _check_int(input.d['a'], flexible_data.object.data['d'].map.data['a'])


class TestFlexibleData2Pyobj(unittest.TestCase):
    def test_flexible_data_2_int(self):
        pass


class TestSerializeAndDeSerialize(unittest.TestCase):
    def test_convert_none(self):
        input = None
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "none convesion error"

    def test_convert_int(self):
        input = 1
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "int convesion error"

    def test_convert_float(self):
        input = 2.012
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert _compare_float(input, output), "float convesion error"

    def test_convert_str(self):
        input = "123"
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "str convesion error"

    def test_convert_bool(self):
        input = False
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "bool convesion error"

    def test_convert_list(self):
        input = [1, 2, 3]
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "list convesion error"

    def test_convert_tuple(self):
        input = (1, 2, 3)
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert list(input) == output, "tuple convesion error"

    def test_convert_set(self):
        input = {1, 2, 3}
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "set convesion error"

    def test_convert_dict(self):
        input = {1: 1, 2: 2, 3: 3}
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "dict convesion error"

    def test_convert_compound_object(self):
        input = {
            False: 1,
            '2': 3,
            4.0: [5, 6.0, ['7', {
                8: 9
            }]],
            'set': {10},
            'dict': {
                11: 12
            }
        }
        flexible_data = module_desc_pb2.FlexibleData()
        from_pyobj_to_flexible_data(input, flexible_data)
        output = from_flexible_data_to_pyobj(flexible_data)
        assert input == output, "dict convesion error"


if __name__ == "__main__":
    unittest.main()
