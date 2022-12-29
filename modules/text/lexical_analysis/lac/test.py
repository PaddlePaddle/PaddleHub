import os
import shutil
import unittest

import paddlehub as hub

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = "今天是个好日子"
        cls.texts = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
        cls.module = hub.Module(name="lac")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('inference')

    def test_cut1(self):
        results = self.module.cut(text=self.text, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, ['今天', '是', '个', '好日子'])

    def test_cut2(self):
        results = self.module.cut(text=self.texts, use_gpu=False, batch_size=1, return_tag=False)
        self.assertEqual(results, [{
            'word': ['今天', '是', '个', '好日子']
        }, {
            'word': ['天气预报', '说', '今天', '要', '下雨']
        }, {
            'word': ['下', '一班', '地铁', '马上', '就要', '到', '了']
        }])

    def test_cut3(self):
        results = self.module.cut(text=self.texts, use_gpu=False, batch_size=2, return_tag=False)
        self.assertEqual(results, [{
            'word': ['今天', '是', '个', '好日子']
        }, {
            'word': ['天气预报', '说', '今天', '要', '下雨']
        }, {
            'word': ['下', '一班', '地铁', '马上', '就要', '到', '了']
        }])

    def test_cut4(self):
        results = self.module.cut(text=self.texts, use_gpu=True, batch_size=2, return_tag=False)
        self.assertEqual(results, [{
            'word': ['今天', '是', '个', '好日子']
        }, {
            'word': ['天气预报', '说', '今天', '要', '下雨']
        }, {
            'word': ['下', '一班', '地铁', '马上', '就要', '到', '了']
        }])

    def test_cut5(self):
        results = self.module.cut(text=self.texts, use_gpu=True, batch_size=2, return_tag=True)
        self.assertEqual(results, [{
            'word': ['今天', '是', '个', '好日子'],
            'tag': ['TIME', 'v', 'q', 'n']
        }, {
            'word': ['天气预报', '说', '今天', '要', '下雨'],
            'tag': ['n', 'v', 'TIME', 'v', 'v']
        }, {
            'word': ['下', '一班', '地铁', '马上', '就要', '到', '了'],
            'tag': ['f', 'm', 'n', 'd', 'v', 'v', 'xc']
        }])

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == '__main__':
    unittest.main()
