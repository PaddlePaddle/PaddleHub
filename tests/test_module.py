# coding=utf-8
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


class TestModule(unittest.TestCase):
    #TODO(ZeyuChen): add setup for test envrinoment prepration
    def test_word2vec_module_usage(self):
        pass
        # url = "http://paddlehub.cdn.bcebos.com/word2vec/word2vec-dim16-simple-example-2.tar.gz"
        # module = Module(module_url=url)
        # inputs = [["it", "is", "new"], ["hello", "world"]]
        # tensor = module._process_input(inputs)
        # print(tensor)
        # result = module(inputs)
        # print(result)

    def test_senta_module_usage(self):
        pass
        # m = Module(module_dir="./models/bow_net")
        # inputs = [["外人", "爸妈", "翻车"], ["金钱", "电量"]]
        # tensor = m._preprocess_input(inputs)
        # print(tensor)
        # result = m({"words": tensor})
        # print(result)


if __name__ == "__main__":
    unittest.main()
