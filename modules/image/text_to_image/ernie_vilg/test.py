import shutil
import unittest

import paddlehub as hub


class TestHubModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = hub.Module(name="ernie_vilg")

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree('ernievilg_output')

    def test_generate_image(self):
        self.module.generate_image(text_prompts=['戴眼镜的猫'],
                                   style="像素风格",
                                   topk=6,
                                   visualization=True,
                                   output_dir='ernievilg_output')


if __name__ == "__main__":
    unittest.main()
