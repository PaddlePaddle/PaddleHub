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
        self.user_dict_path = '../lac_user_dict/user.dict'
        self.test_text = ["今天是个好日子", "调料份量不能多，也不能少，味道才能正好"]
        self.results_tag = [
            {
                'word': ['今天', '是', '个', '好日子'],
                'tag': ['TIME', 'v', 'q', 'n']
            },
            {
                'word':['调料', '份量', '不能', '多', '，', '也', '不能少', '，', '味道', '才能', '正好'],
                'tag': ['n', 'n', 'v', 'a', 'w', 'd', 'a', 'w', 'n', 'v', 'd']
            }
        ]
        self.results_notag = [
            {
                'word': ['今天', '是', '个', '好日子']
            },
            {
                'word': ['调料', '份量', '不能', '多', '，', '也', '不能少', '，', '味道', '才能', '正好']
            }
        ]
        self.results_notag_userdict = [
            {
                'word': ['今天', '是', '个', '好日子']
            },
            {
                'word': ['调料', '份量', '不能', '多', '，', '也', '不能', '少', '，', '味道', '才能', '正好']
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
        self.assertNotEqual(self.module.interventer, None)

    def test_del_user_dict(self):
        self.module.set_user_dict(self.user_dict_path)
        self.assertNotEqual(self.module.interventer, None)
        self.module.del_user_dict()
        self.assertEqual(self.module.interventer, None)

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

    def test_get_tags(self):
        tags = self.module.get_tags()
        self.assertEqual(tags, self.tags)


if __name__ == '__main__':
    main()
