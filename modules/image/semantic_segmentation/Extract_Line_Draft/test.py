import os
import shutil
import unittest

import cv2
import requests
import paddlehub as hub


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class TestHubModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        img_url = 'https://ai-studio-static-online.cdn.bcebos.com/1c30757e069541a18dc89b92f0750983b77ad762560849afa0170046672e57a3'
        if not os.path.exists('tests'):
            os.makedirs('tests')
        response = requests.get(img_url)
        assert response.status_code == 200, 'Network Error.'
        with open('tests/test.jpg', 'wb') as f:
            f.write(response.content)
        cls.module = hub.Module(name="Extract_Line_Draft")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('tests')
        shutil.rmtree('inference')
        shutil.rmtree('output')

    def test_ExtractLine1(self):
        self.module.ExtractLine(
            image='tests/test.jpg',
            use_gpu=False
        )
        self.assertTrue(os.path.exists('output/output.png'))

    def test_ExtractLine2(self):
        self.module.ExtractLine(
            image='tests/test.jpg',
            use_gpu=True
        )
        self.assertTrue(os.path.exists('output/output.png'))

    def test_ExtractLine3(self):
        self.assertRaises(
            AttributeError,
            self.module.ExtractLine,
            image='no.jpg'
        )

    def test_ExtractLine4(self):
        self.assertRaises(
            TypeError,
            self.module.ExtractLine,
            image=['tests/test.jpg']
        )

    def test_save_inference_model(self):
        self.module.save_inference_model('./inference/model')

        self.assertTrue(os.path.exists('./inference/model.pdmodel'))
        self.assertTrue(os.path.exists('./inference/model.pdiparams'))


if __name__ == "__main__":
    unittest.main()
