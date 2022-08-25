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

        data = results[0]['data']
        self.assertEqual(len(data), 1)

if __name__ == "__main__":
    unittest.main()