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
import paddle_hub as hub
import paddle.fluid as fluid
from paddle_hub import create_signature


class TestSignature(unittest.TestCase):
    def test_check_signature_info(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            var_1 = fluid.layers.data(name="var_1", dtype="int64", shape=[1])
            var_2 = fluid.layers.data(
                name="var_2", dtype="float32", shape=[3, 100, 100])
            name = "test"
            inputs = [var_1]
            outputs = [var_2]
            feed_names = ["label"]
            fetch_names = ["img"]
            sign = create_signature(
                name=name,
                inputs=inputs,
                outputs=outputs,
                feed_names=feed_names,
                fetch_names=fetch_names)
            assert sign.get_name() == name, "sign name error"
            assert sign.get_inputs() == inputs, "sign inputs error"
            assert sign.get_outputs() == outputs, "sign outputs error"
            assert sign.get_feed_names() == feed_names, "sign feed_names error"
            assert sign.get_fetch_names(
            ) == fetch_names, "sign fetch_names error"


if __name__ == "__main__":
    unittest.main()
