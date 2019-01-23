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
import paddle.fluid as fluid


class TestModule(unittest.TestCase):
    #TODO(ZeyuChen): add setup for test envrinoment prepration
    def test_word2vec_module_usage(self):
        url = "https://paddlehub.cdn.bcebos.com/word2vec/word2vec_test_module.tar.gz"
        w2v_module = hub.Module(module_url=url)
        feed_dict, fetch_dict, program = w2v_module(
            sign_name="default", trainable=False)
        with fluid.program_guard(main_program=program):
            pred_prob = fetch_dict["pred_prob"]
            pred_word = fluid.layers.argmax(x=pred_prob, axis=1)
            # set place, executor, datafeeder
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            feed_vars = [
                feed_dict["firstw"], feed_dict["secondw"], feed_dict["thirdw"],
                feed_dict["fourthw"]
            ]
            feeder = fluid.DataFeeder(place=place, feed_list=feed_vars)

            word_ids = [[1, 2, 3, 4]]
            result = exe.run(
                fluid.default_main_program(),
                feed=feeder.feed(word_ids),
                fetch_list=[pred_word],
                return_numpy=True)

            self.assertEqual(result[0], 5)

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
