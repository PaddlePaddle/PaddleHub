import paddle
import paddlehub as hub
import numpy as np
import os
from paddlehub import BaseProcessor
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax


def draw_bounding_box_on_image(image_path, data_list, save_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for data in data_list:
        left, right, top, bottom = data['left'], data['right'], data[
            'top'], data['bottom']
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=4,
                  fill='red')
        if image.mode == 'RGB':
            draw.text((left, top), data['label'], (255, 255, 0))

    image_name = image_path.split('/')[-1]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, image_name)
    print("image with bbox drawed saved as {}".format(save_path))
    image.save(save_path)


class Processor(BaseProcessor):
    def __init__(self, module):
        self.module = module
        label_list_file = os.path.join(self.module.helper.assets_path(),
                                       "label_list.txt")
        with open(label_list_file, "r") as file:
            content = file.read()
        self.label_list = content.split("\n")
        self.confs_threshold = 0.5

    def preprocess(self, sign_name, data_dict):
        def process_image(img):
            if img.mode == 'L':
                img = im.convert('RGB')
            im_width, im_height = img.size
            img = img.resize((300, 300), Image.ANTIALIAS)
            img = np.array(img)
            # HWC to CHW
            if len(img.shape) == 3:
                img = np.swapaxes(img, 1, 2)
                img = np.swapaxes(img, 1, 0)
            # RBG to BGR
            img = img[[2, 1, 0], :, :]
            img = img.astype('float32')
            mean_value = [127.5, 127.5, 127.5]
            mean_value = np.array(mean_value)[:, np.newaxis, np.newaxis].astype(
                'float32')
            img -= mean_value
            img = img * 0.007843
            return img

        result = {'image': []}
        for path in data_dict['image']:
            img = Image.open(path)
            im_width, im_height = img.size
            result_i = {}
            result_i['path'] = path
            result_i['width'] = im_width
            result_i['height'] = im_height
            result_i['processed'] = process_image(img)
            result['image'].append(result_i)
        return result

    def postprocess(self, sign_name, data_out, data_info, **kwargs):
        if sign_name == "object_detection":
            lod_tensor = data_out[0]
            lod = lod_tensor.lod()[0]
            results = np.array(data_out[0])
            output = []
            for index in range(len(lod) - 1):
                result_i = results[lod[index]:lod[index + 1]]
                output_i = {
                    'path': data_info['image'][index]['path'],
                    'data': []
                }
                for dt in result_i:
                    if dt[1] < self.confs_threshold:
                        continue
                    dt_i = {}
                    category_id = dt[0]
                    bbox = dt[2:]
                    xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
                    (left, right, top,
                     bottom) = (xmin * data_info['image'][index]['width'],
                                xmax * data_info['image'][index]['width'],
                                ymin * data_info['image'][index]['height'],
                                ymax * data_info['image'][index]['height'])
                    dt_i['left'] = left
                    dt_i['right'] = right
                    dt_i['top'] = top
                    dt_i['bottom'] = bottom
                    dt_i['label'] = self.label_list[int(category_id)]
                    output_i['data'].append(dt_i)
                draw_bounding_box_on_image(
                    output_i['path'], output_i['data'], save_path="test_result")
                output.append(output_i)

            return output

    def data_format(self, sign_name):
        if sign_name == "object_detection":
            return {
                "image": {
                    'type': hub.DataType.IMAGE,
                    'feed_key': self.module.signatures[sign_name].inputs[0].name
                }
            }
        return None
