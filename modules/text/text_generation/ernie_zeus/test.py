import unittest

import paddlehub as hub


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = hub.Module(name='ernie_zeus')

    def test_custom_generation(self):
        results = self.module.custom_generation('你好，')
        self.assertIsInstance(results, str)

    def test_text_generation(self):
        results = self.module.text_generation('给宠物猫起一些可爱的名字。名字：')
        self.assertIsInstance(results, str)

    def test_text_summarization(self):
        results = self.module.text_summarization(
            '在芬兰、瑞典提交“入约”申请近一个月来，北约成员国内部尚未对此达成一致意见。与此同时，俄罗斯方面也多次对北约“第六轮扩张”发出警告。据北约官网显示，北约秘书长斯托尔滕贝格将于本月12日至13日出访瑞典和芬兰，并将分别与两国领导人进行会晤。'
        )
        self.assertIsInstance(results, str)

    def test_copywriting_generation(self):
        results = self.module.copywriting_generation('芍药香氛的沐浴乳')
        self.assertIsInstance(results, str)

    def test_modulenovel_continuation(self):
        results = self.module.novel_continuation('昆仑山可以说是天下龙脉的根源，所有的山脉都可以看作是昆仑的分支。这些分出来的枝枝杈杈，都可以看作是一条条独立的龙脉。')
        self.assertIsInstance(results, str)

    def test_answer_generation(self):
        results = self.module.answer_generation('交朋友的原则是什么？')
        self.assertIsInstance(results, str)

    def test_couplet_continuation(self):
        results = self.module.couplet_continuation('五湖四海皆春色')
        self.assertIsInstance(results, str)

    def test_composition_generation(self):
        results = self.module.composition_generation('诚以养德，信以修身')
        self.assertIsInstance(results, str)

    def test_text_cloze(self):
        results = self.module.text_cloze('她有着一双[MASK]的眼眸。')
        self.assertIsInstance(results, str)


if __name__ == "__main__":
    unittest.main()
