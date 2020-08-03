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
from unittest import TestCase, main
import paddlehub as hub


class ErnieGenCoupletTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='ernie_gen_couplet')
        self.left = ["风吹云乱天垂泪", "若有经心风过耳"]

    def test_predict(self):
        rights = self.module.generate(self.left)
        self.assertEqual(len(rights), 2)
        self.assertEqual(len(rights[0]), 5)
        self.assertEqual(len(rights[0][0]), 7)
        self.assertEqual(len(rights[1][0]), 7)


if __name__ == '__main__':
    main()
