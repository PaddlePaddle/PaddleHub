import argparse
import base64
import os
import time
from typing import Union

import cv2
import numpy as np
import paddle
import paddle.nn as nn

from .fbcnn import FBCNN
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tobytes()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return data


@moduleinfo(
    name='fbcnn_gray',
    version='1.0.0',
    type="CV/image_editing",
    author="",
    author_email="",
    summary="Flexible JPEG Artifacts Removal.",
)
class FBCNNGary(nn.Layer):

    def __init__(self):
        super(FBCNNGary, self).__init__()
        self.default_pretrained_model_path = os.path.join(self.directory, 'ckpts', 'fbcnn_gray.pdparams')
        self.fbcnn = FBCNN(in_nc=1, out_nc=1)
        state_dict = paddle.load(self.default_pretrained_model_path)
        self.fbcnn.set_state_dict(state_dict)
        self.fbcnn.eval()

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = img[None, ...]
        img = img / 255.0
        return img.astype(np.float32)

    def postprocess(self, img: np.ndarray) -> np.ndarray:
        img = img.clip(0, 1)
        img = img * 255.0
        return img.astype(np.uint8)

    def artifacts_removal(self,
                          image: Union[str, np.ndarray],
                          quality_factor: float = None,
                          visualization: bool = True,
                          output_dir: str = "fbcnn_gray_output") -> np.ndarray:
        if isinstance(image, str):
            _, file_name = os.path.split(image)
            save_name, _ = os.path.splitext(file_name)
            save_name = save_name + '_' + str(int(time.time())) + '.jpg'
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        elif isinstance(image, np.ndarray):
            save_name = str(int(time.time())) + '.jpg'
            image = image
        else:
            raise Exception("image should be a str / np.ndarray")

        with paddle.no_grad():
            img_input = self.preprocess(image)
            img_input = paddle.to_tensor(img_input[None, ...], dtype=paddle.float32)
            if quality_factor and 0 <= quality_factor <= 1:
                qf_input = paddle.to_tensor([[quality_factor]], dtype=paddle.float32)
            else:
                qf_input = None
            img_output, _ = self.fbcnn(img_input, qf_input)
            img_output = img_output.numpy()[0][0]
            img_output = self.postprocess(img_output)

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
        self.parser.add_argument('--quality_factor', type=float, default=None, help="Image quality factor (0.0-1.0).")
        self.parser.add_argument('--output_dir',
                                 type=str,
                                 default='fbcnn_gray_output',
                                 help="The directory to save output images.")
        args = self.parser.parse_args(argvs)
        self.artifacts_removal(image=args.input_path,
                               quality_factor=args.quality_factor,
                               visualization=True,
                               output_dir=args.output_dir)
        return 'Artifacts removal results are saved in %s' % args.output_dir

    @serving
    def serving_method(self, image, **kwargs):
        """
        Run as a service.
        """
        image = base64_to_cv2(image)
        img_output = self.artifacts_removal(image=image, **kwargs)

        return cv2_to_base64(img_output)
