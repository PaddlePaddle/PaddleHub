from __future__ import absolute_import
from __future__ import division

import os
import cv2
import argparse
import base64
import paddlex as pdx

import numpy as np
import paddlehub as hub
from paddlehub.module.module import moduleinfo, runnable, serving


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def cv2_to_base64(image):
    # return base64.b64encode(image)
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def read_images(paths):
    images = []
    for path in paths:
        images.append(cv2.imread(path))
    return images


@moduleinfo(
    name='DriverStatusRecognition',
    type='cv/classification',
    author='郑博培、彭兆帅',
    author_email='2733821739@qq.com',
    summary="Distinguish the driver's normal driving, making a phone call, drinking water and other different actions",
    version='1.0.0')
class MODULE(hub.Module):
    def _initialize(self, **kwargs):
        self.default_pretrained_model_path = os.path.join(self.directory, 'assets')
        self.model = pdx.deploy.Predictor(self.default_pretrained_model_path, **kwargs)

    def predict(self, images=None, paths=None, data=None, batch_size=1, use_gpu=False, **kwargs):

        all_data = images if images is not None else read_images(paths)
        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = []
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except IndexError:
                    break
            out = self.model.batch_predict(batch_data, **kwargs)
            res.extend(out)
        return res

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        res = []
        for result in results:
            if isinstance(result, dict):
                # result_new = dict()
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        result[key] = cv2_to_base64(value)
                    elif isinstance(value, np.generic):
                        result[key] = np.asscalar(value)

            elif isinstance(result, list):
                for index in range(len(result)):
                    for key, value in result[index].items():
                        if isinstance(value, np.ndarray):
                            result[index][key] = cv2_to_base64(value)
                        elif isinstance(value, np.generic):
                            result[index][key] = np.asscalar(value)
            else:
                raise RuntimeError('The result cannot be used in serving.')
            res.append(result)
        return res

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
            usage='%(prog)s',
            add_help=True)
        self.arg_input_group = self.parser.add_argument_group(title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options", description="Run configuration for controlling module behavior, not required.")
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(paths=[args.input_path], use_gpu=args.use_gpu)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument('--use_gpu', type=bool, default=False, help="whether use GPU or not")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


if __name__ == '__main__':
    module = MODULE(directory='./new_model')
    images = [cv2.imread('./cat.jpg'), cv2.imread('./cat.jpg'), cv2.imread('./cat.jpg')]
    res = module.predict(images=images)
