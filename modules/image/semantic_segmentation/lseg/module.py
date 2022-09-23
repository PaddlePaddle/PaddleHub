import argparse
import base64
import os
import time
from typing import Dict
from typing import List
from typing import Union

import cv2
import numpy as np
import paddle
import paddle.vision.transforms as transforms
from paddlenlp.transformers.clip.tokenizer import CLIPTokenizer

import paddlehub as hub
from . import models
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


@moduleinfo(
    name='lseg',
    version='1.0.0',
    type="CV/semantic_segmentation",
    author="",
    author_email="",
    summary="Language-driven Semantic Segmentation.",
)
class LSeg(models.LSeg):

    def __init__(self):
        super(LSeg, self).__init__()
        self.default_pretrained_model_path = os.path.join(self.directory, 'ckpts', 'LSeg.pdparams')
        state_dict = paddle.load(self.default_pretrained_model_path)
        self.set_state_dict(state_dict)
        self.eval()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

        self.language_recognition = hub.Module(name='baidu_language_recognition')
        self.translate = hub.Module(name='baidu_translate')

    @staticmethod
    def get_colormap(n):
        assert n <= 256, "num_class should be less than 256."

        pallete = [0] * (256 * 3)

        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3

        return np.asarray(pallete, dtype=np.uint8).reshape(256, 1, 3)

    def segment(self,
                image: Union[str, np.ndarray],
                labels: Union[str, List[str]],
                visualization: bool = False,
                output_dir: str = 'lseg_output') -> Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]:
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray):
            image = image
        else:
            raise Exception("image should be a str / np.ndarray")

        if isinstance(labels, str):
            labels = [labels, 'other']
            print('"other" category label is automatically added because the length of labels is equal to 1')
            print('new labels: ', labels)
        elif isinstance(labels, list):
            if len(labels) == 1:
                labels.append('other')
                print('"other" category label is automatically added because the length of labels is equal to 1')
                print('new labels: ', labels)
            elif len(labels) == 0:
                raise Exception("labels should not be empty.")
        else:
            raise Exception("labels should be a str or list.")

        class_num = len(labels)

        labels_ = list(set(labels))
        labels_.sort(key=labels.index)
        labels = labels_

        input_labels = []
        for label in labels:
            from_lang = self.language_recognition.recognize(query=label)
            if from_lang != 'en':
                label = self.translate.translate(query=label, from_lang=from_lang, to_lang='en')
            input_labels.append(label)

        input_labels_ = list(set(input_labels))
        input_labels_.sort(key=input_labels.index)
        input_labels = input_labels_

        if len(input_labels) < class_num:
            print('remove the same labels...')
            print('new labels: ', input_labels)

        h, w = image.shape[:2]
        image = image[:-(h % 32) if h % 32 else None, :-(w % 32) if w % 32 else None]
        images = self.transforms(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        texts = self.tokenizer(input_labels, padding=True, return_tensors="pd")['input_ids']

        with paddle.no_grad():
            results = self.forward(images, texts)
            results = paddle.argmax(results, 1).cast(paddle.uint8)
            gray_seg = results.numpy()[0]

        colormap = self.get_colormap(len(labels))
        color_seg = cv2.applyColorMap(gray_seg, colormap)
        mix_seg = cv2.addWeighted(image, 0.5, color_seg, 0.5, 0.0)

        classes_seg = {}
        for i, label in enumerate(input_labels):
            mask = (gray_seg == i).astype('uint8')
            classes_seg[label] = cv2.bitwise_and(image, image, mask=mask)

        if visualization:
            save_dir = os.path.join(output_dir, str(int(time.time())))
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            for label, dst in classes_seg.items():
                cv2.imwrite(os.path.join(save_dir, '%s.jpg' % label), dst)

        return {'gray': gray_seg, 'color': color_seg, 'mix': mix_seg, 'classes': classes_seg}

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.parser.add_argument('--input_path', type=str, help="path to image.")
        self.parser.add_argument('--labels', type=str, nargs='+', help="segmentation labels.")
        self.parser.add_argument('--output_dir',
                                 type=str,
                                 default='lseg_output',
                                 help="The directory to save output images.")
        args = self.parser.parse_args(argvs)
        self.segment(image=args.input_path, labels=args.labels, visualization=True, output_dir=args.output_dir)
        return 'segmentation results are saved in %s' % args.output_dir

    @serving
    def serving_method(self, image, **kwargs):
        """
        Run as a service.
        """
        image = base64_to_cv2(image)
        results = self.segment(image=image, **kwargs)

        return {
            'gray': cv2_to_base64(results['gray']),
            'color': cv2_to_base64(results['color']),
            'mix': cv2_to_base64(results['mix']),
            'classes': {k: cv2_to_base64(v)
                        for k, v in results['classes'].items()}
        }
