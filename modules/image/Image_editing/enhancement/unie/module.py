import argparse
import base64
import os
import time
from typing import Union

import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms

from .unie import ResnetGenerator
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
    name='unie',
    version='1.0.0',
    type="CV/image_editing",
    author="",
    author_email="",
    summary="Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression.",
)
class UNIE(nn.Layer):

    def __init__(self):
        super(UNIE, self).__init__()
        self.default_pretrained_model_path = os.path.join(self.directory, 'LOL_params_0900000.pdparams')
        self.gan = ResnetGenerator(3, 3)
        state_dict = paddle.load(self.default_pretrained_model_path)
        self.gan.set_state_dict(state_dict)
        self.gan.eval()
        self.img_size = 512
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        return img, h, w

    def postprocess(self, img: np.ndarray, h, w) -> np.ndarray:
        img = img * 0.5 + 0.5
        img = img * 255.0
        img = img.clip(0, 255)
        img = img.transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (w, h))
        return img.astype(np.uint8)

    def night_enhancement(self,
                          image: Union[str, np.ndarray],
                          visualization: bool = True,
                          output_dir: str = "unie_output") -> np.ndarray:
        if isinstance(image, str):
            _, file_name = os.path.split(image)
            save_name, _ = os.path.splitext(file_name)
            save_name = save_name + '_' + str(int(time.time())) + '.jpg'
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            save_name = str(int(time.time())) + '.jpg'
            image = image
        else:
            raise Exception("image should be a str / np.ndarray")

        with paddle.no_grad():
            img_input, h, w = self.preprocess(image)
            img_input = paddle.to_tensor(self.test_transform(img_input)[None, ...], dtype=paddle.float32)

            img_output, _, _ = self.gan(img_input)
            img_output = img_output.numpy()[0]
            img_output = self.postprocess(img_output, h, w)

        if visualization:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            save_path = os.path.join(output_dir, save_name)
            cv2.imwrite(save_path, img_output)

        return img_output

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(description="Run the {} module.".format(self.name),
                                              prog='hub run {}'.format(self.name),
                                              usage='%(prog)s',
                                              add_help=True)
        self.parser.add_argument('--input_path', type=str, help="Path to image.")
        self.parser.add_argument('--output_dir',
                                 type=str,
                                 default='unie_output',
                                 help="The directory to save output images.")
        args = self.parser.parse_args(argvs)
        self.night_enhancement(image=args.input_path, visualization=True, output_dir=args.output_dir)
        return 'Results are saved in %s' % args.output_dir

    @serving
    def serving_method(self, image, **kwargs):
        """
        Run as a service.
        """
        image = base64_to_cv2(image)
        img_output = self.night_enhancement(image=image, **kwargs)

        return cv2_to_base64(img_output)
