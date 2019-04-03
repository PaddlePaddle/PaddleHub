import os

import paddle
import numpy as np
from PIL import Image

from paddlehub import BaseProcessor
import paddlehub as hub

DATA_DIM = 224
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def softmax(x):
    orig_shape = x.shape

    if len(x.shape) > 1:
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img):
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std

    return img


class Processor(BaseProcessor):
    def __init__(self, module):
        self.module = module
        label_list_file = os.path.join(self.module.helper.assets_path(),
                                       "label_list.txt")
        with open(label_list_file, "r") as file:
            content = file.read()
        self.label_list = content.split("\n")

    def build_config(self, **kwargs):
        self.top_only = kwargs.get("top_only", None)
        try:
            self.top_only = bool(self.top_only)
        except:
            self.top_only = False

    def preprocess(self, sign_name, data_dict):
        result = {'image': []}
        for path in data_dict['image']:
            result_i = {}
            result_i['processed'] = process_image(Image.open(path))
            result['image'].append(result_i)
        return result

    def postprocess(self, sign_name, data_out, data_info, **kwargs):
        self.build_config(**kwargs)
        if sign_name == "classification":
            results = np.array(data_out[0])
            output = []
            for index, result in enumerate(results):
                result_i = softmax(result)
                if self.top_only:
                    index = np.argsort(result_i)[::-1][:1][0]
                    label = self.label_list[index]
                    output.append({label: result_i[index]})
                else:
                    output.append({
                        self.label_list[index]: value
                        for index, value in enumerate(result_i)
                    })
            return [output]
        elif sign_name == "feature_map":
            return np.array(results)

    def data_format(self, sign_name):
        if sign_name == "classification":
            return {
                "image": {
                    'type': hub.DataType.IMAGE,
                    'feed_key': self.module.signatures[sign_name].inputs[0].name
                }
            }
        elif sign_name == "feature_map":
            return {
                "image": {
                    'type': hub.DataType.IMAGE,
                    'feed_key': self.module.signatures[sign_name].inputs[0].name
                }
            }
