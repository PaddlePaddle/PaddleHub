import os
from unittest import TestCase, main
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlehub as hub


class ReadingPicturesWritingPoemsTestCase(TestCase):
    def setUp(self):
        self.module = hub.Module(name='reading_pictures_writing_poems')
        self.test_image = "castle.jpg"
        self.results = [{
            'image': 'castle.jpg',
            'Poetrys': '山川山陵山，沟渠村庄沟。我来春雨余，草木亦已柔。'
        }]
        

    def test_writing_poems(self):
        # test gpu
        results = self.module.WritingPoem(
            image=self.test_image, use_gpu=True)
        self.assertEqual(results, self.results)

if __name__ == '__main__':
    main()