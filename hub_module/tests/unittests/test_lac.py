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
import os
from unittest import TestCase, main
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlehub as hub


class LacTestCase(TestCase):
    # yapf: disable
    def setUp(self):
        self.module = hub.Module(name='lac')
        self.user_dict_path = '../user.dict'
        self.test_text = ["今天是个好日子", "春天的花开秋天的风以及冬天的落阳"]
        self.results_tag = [
            {
                'word': ['今天', '是', '个', '好日子'],
                'tag': ['TIME', 'v', 'q', 'n']
            },
            {
                'word': ['春天', '的', '花开', '秋天', '的', '风', '以及', '冬天', '的', '落阳'],
                'tag': ['TIME', 'u', 'v', 'TIME', 'u', 'n', 'c', 'TIME', 'u', 'vn']
            }
        ]
        self.results_notag = [
            {
                'word': ['今天', '是', '个', '好日子']
            },
            {
                'word': ['春天', '的', '花开', '秋天', '的', '风', '以及', '冬天', '的', '落阳']
            }
        ]
        self.results_notag_userdict = [
            {
                'word': ['今天', '是', '个', '好日子']
            },
            {
                'word':  ['春天', '的', '花', '开', '秋天的风', '以及', '冬天', '的', '落', '阳']
            }
        ]
        self.tags = {
            'n': '普通名词',
            'f': '方位名词',
            's': '处所名词',
            't': '时间',
            'nr': '人名',
            'ns': '地名',
            'nt': '机构名',
            'nw': '作品名',
            'nz': '其他专名',
            'v': '普通动词',
            'vd': '动副词',
            'vn': '名动词',
            'a': '形容词',
            'ad': '副形词',
            'an': '名形词',
            'd': '副词',
            'm': '数量词',
            'q': '量词',
            'r': '代词',
            'p': '介词',
            'c': '连词',
            'u': '助词',
            'xc': '其他虚词',
            'w': '标点符号',
            'PER': '人名',
            'LOC': '地名',
            'ORG': '机构名',
            'TIME': '时间'
        }
    # yapf: enable.

    def test_set_user_dict(self):
        self.module.set_user_dict(self.user_dict_path)
        self.assertNotEqual(self.module.custom, None)

    def test_del_user_dict(self):
        self.module.set_user_dict(self.user_dict_path)
        self.assertNotEqual(self.module.custom, None)
        self.module.del_user_dict()
        self.assertEqual(self.module.custom, None)

    def test_lexical_analysis(self):
        self.module.del_user_dict()

        # test batch_size
        results = self.module.lexical_analysis(
            texts=self.test_text, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag)
        results = self.module.lexical_analysis(
            texts=self.test_text,
            use_gpu=False,
            batch_size=10,
            return_tag=False)
        self.assertEqual(results, self.results_notag)

        # test return_tag
        results = self.module.lexical_analysis(
            texts=self.test_text, use_gpu=False, batch_size=1, return_tag=True)
        self.assertEqual(results, self.results_tag)

        # test gpu
        results = self.module.lexical_analysis(
            texts=self.test_text, use_gpu=True, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag)

        # test results to add user_dict
        self.module.set_user_dict(self.user_dict_path)
        results = self.module.lexical_analysis(
            texts=self.test_text, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag_userdict)

    def test_cut(self):
        # test batch_size
        results = self.module.cut(
            text=self.test_text, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag)
        results = self.module.cut(
            text=self.test_text, use_gpu=False, batch_size=10, return_tag=False)
        self.assertEqual(results, self.results_notag)

        # test return_tag
        results = self.module.cut(
            text=self.test_text, use_gpu=False, batch_size=1, return_tag=True)
        self.assertEqual(results, self.results_tag)

        # test gpu
        results = self.module.cut(
            text=self.test_text, use_gpu=True, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag)

        # test results to add user_dict
        self.module.set_user_dict(self.user_dict_path)
        results = self.module.cut(
            text=self.test_text, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, self.results_notag_userdict)

        # test single sentence
        results = self.module.cut(
            text="今天是个好日子", use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, ['今天', '是', '个', '好日子'])

    def test_get_tags(self):
        tags = self.module.get_tags()
        self.assertEqual(tags, self.tags)


if __name__ == '__main__':
    main()
