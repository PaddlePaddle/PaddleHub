import unittest
import paddlehub as hub


class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = hub.Module(name="faster_rcnn_resnet50_coco2017")

    def test_object_detection(self):
        results = self.module.object_detection(
            paths=['test.jpg'],
            visualization=False
        )
        bbox = results[0]['data'][0]
        label = bbox['label']
        confidence = bbox['confidence']
        left = bbox['left']
        right = bbox['right']
        top = bbox['top']
        bottom = bbox['bottom']

        self.assertEqual(label, 'cat')
        self.assertTrue(confidence > 0.5)
        self.assertTrue(450 < left < 550)
        self.assertTrue(2850 < right < 2950)
        self.assertTrue(750 < top < 850)
        self.assertTrue(4000 < bottom < 4100)


if __name__ == "__main__":
    unittest.main()
