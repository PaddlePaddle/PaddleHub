# coding=utf-8
import unittest
import paddlehub as hub


class TestVideoTag(unittest.TestCase):
    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.module = hub.Module(name='videotag_tsn_lstm')
        self.test_video = [
            "../video_dataset/classification/1.mp4",
            "../video_dataset/classification/2.mp4"
        ]

    def test_classification(self):
        default_expect1 = {
            '训练': 0.9771281480789185,
            '蹲': 0.9389840960502625,
            '杠铃': 0.8554490804672241,
            '健身房': 0.8479971885681152
        }
        default_expect2 = {'舞蹈': 0.8504238724708557}
        for use_gpu in [True, False]:
            for threshold in [0.5, 0.9]:
                for top_k in [10, 1]:
                    expect1 = {}
                    expect2 = {}
                    for key, value in default_expect1.items():
                        if value >= threshold:
                            expect1[key] = value
                        if len(expect1.keys()) >= top_k:
                            break
                    for key, value in default_expect2.items():
                        if value >= threshold:
                            expect2[key] = value
                        if len(expect2.keys()) >= top_k:
                            break
                    results = self.module.classify(
                        paths=self.test_video,
                        use_gpu=use_gpu,
                        threshold=threshold,
                        top_k=top_k)
                    for result in results:
                        if result['path'] == '1.mp4':
                            self.assertEqual(result['prediction'], expect1)
                        else:
                            self.assertEqual(result['prediction'], expect2)


if __name__ == "__main__":
    unittest.main()
