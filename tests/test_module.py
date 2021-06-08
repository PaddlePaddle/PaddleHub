# coding=utf-8
import os
import unittest
import paddlehub as hub


class TestHubModule(unittest.TestCase):
    def test_lac(self):
        lac = hub.Module(name="lac")
        test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
        inputs = {"text": test_text}
        results = lac.lexical_analysis(data=inputs)
        self.assertEqual(results[0]['word'], ['今天', '是', '个', '好日子'])
        self.assertEqual(results[0]['tag'], ['TIME', 'v', 'q', 'n'])
        self.assertEqual(results[1]['word'], ['天气预报', '说', '今天', '要', '下雨'])
        self.assertEqual(results[1]['tag'], ['n', 'v', 'TIME', 'v', 'v'])
        self.assertEqual(results[2]['word'], ['下', '一班', '地铁', '马上', '就要', '到', '了'])
        self.assertEqual(results[2]['tag'], ['f', 'm', 'n', 'd', 'v', 'v', 'xc'])

    def test_senta(self):
        senta = hub.Module(name="senta_bilstm")
        test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
        input_dict = {"text": test_text}
        results = senta.sentiment_classify(data=input_dict)
        self.assertEqual(results[0]['sentiment_label'], 1)
        self.assertEqual(results[0]['sentiment_key'], 'positive')
        self.assertEqual(results[1]['sentiment_label'], 0)
        self.assertEqual(results[1]['sentiment_key'], 'negative')
        for result in results:
            print(result['text'])
            print(result['positive_probs'])
            print(result['negative_probs'])

    def test_simnet(self):
        simnet_bow = hub.Module(name="simnet_bow")
        test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
        test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
        inputs = {"text_1": test_text_1, "text_2": test_text_2}
        results = simnet_bow.similarity(data=inputs)
        max_score = -1
        result_text = ""
        for result in results:
            if result['similarity'] > max_score:
                max_score = result['similarity']
                result_text = result['text_2']
        print("The most matching with the %s is %s" % (test_text_1[0], result_text))

    def test_ssd(self):
        ssd = hub.Module(name="ssd_mobilenet_v1_pascal")
        test_img_path = os.path.join(os.path.dirname(__file__), "resources", "test_img_cat.jpg")
        input_dict = {"image": [test_img_path]}
        results = ssd.object_detection(data=input_dict)
        for result in results:
            print(result['path'])
            print(result['data'])


if __name__ == "__main__":
    unittest.main()
