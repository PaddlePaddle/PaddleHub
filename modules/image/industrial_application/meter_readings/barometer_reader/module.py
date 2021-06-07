import math
import os
import base64
from typing import Union

import argparse
import cv2
import numpy as np
import paddlehub as hub
import paddlex as pdx
import paddle.nn as nn
from paddlex.seg import transforms as T
from paddlehub.module.module import moduleinfo, runnable, serving

METER_SHAPE = 512
CIRCLE_CENTER = [256, 256]
CIRCLE_RADIUS = 250
PI = 3.1415926536
LINE_HEIGHT = 120
LINE_WIDTH = 1570
TYPE_THRESHOLD = 40
METER_CONFIG = [{
    'scale_value': 25.0 / 50.0,
    'range': 25.0,
    'unit': "(MPa)"
}, {
    'scale_value': 1.6 / 32.0,
    'range': 1.6,
    'unit': "(MPa)"
}]


def base64_to_cv2(b64str: str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def cv2_to_base64(image: np.ndarray):
    # return base64.b64encode(image)
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


@moduleinfo(
    name="barometer_reader",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary=
    "meter_reader implements the detection and automatic reading of traditional mechanical pointer meters based on Meter detection and  pointer segmentation.",
    version="1.0.0")
class BarometerReader(nn.Layer):
    def __init__(self):
        super(BarometerReader, self).__init__()
        self.detector = pdx.load_model(os.path.join(self.directory, 'meter_det_inference_model'))
        self.segmenter = pdx.load_model(os.path.join(self.directory, 'meter_seg_inference_model'))
        self.seg_transform = T.Compose([T.Normalize()])

    def read_process(self, label_maps: np.ndarray):
        line_images = self.creat_line_image(label_maps)
        scale_data, pointer_data = self.convert_1d_data(line_images)
        self.scale_mean_filtration(scale_data)
        result = self.get_meter_reader(scale_data, pointer_data)
        return result

    def creat_line_image(self, meter_image: np.ndarray):
        line_image = np.zeros((LINE_HEIGHT, LINE_WIDTH), dtype=np.uint8)
        for row in range(LINE_HEIGHT):
            for col in range(LINE_WIDTH):
                theta = PI * 2 / LINE_WIDTH * (col + 1)
                rho = CIRCLE_RADIUS - row - 1
                x = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
                y = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
                line_image[row, col] = meter_image[x, y]
        return line_image

    def convert_1d_data(self, meter_image: np.ndarray):
        scale_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        pointer_data = np.zeros((LINE_WIDTH), dtype=np.uint8)
        for col in range(LINE_WIDTH):
            for row in range(LINE_HEIGHT):
                if meter_image[row, col] == 1:
                    pointer_data[col] += 1
                elif meter_image[row, col] == 2:
                    scale_data[col] += 1
        return scale_data, pointer_data

    def scale_mean_filtration(self, scale_data: np.ndarray):
        mean_data = np.mean(scale_data)
        for col in range(LINE_WIDTH):
            if scale_data[col] < mean_data:
                scale_data[col] = 0

    def get_meter_reader(self, scale_data: np.ndarray, pointer_data: np.ndarray):
        scale_flag = False
        pointer_flag = False
        one_scale_start = 0
        one_scale_end = 0
        one_pointer_start = 0
        one_pointer_end = 0
        scale_location = list()
        pointer_location = 0
        for i in range(LINE_WIDTH - 1):
            if scale_data[i] > 0 and scale_data[i + 1] > 0:
                if scale_flag == False:
                    one_scale_start = i
                    scale_flag = True
            if scale_flag:
                if scale_data[i] == 0 and scale_data[i + 1] == 0:
                    one_scale_end = i - 1
                    one_scale_location = (one_scale_start + one_scale_end) / 2
                    scale_location.append(one_scale_location)
                    one_scale_start = 0
                    one_scale_end = 0
                    scale_flag = False
            if pointer_data[i] > 0 and pointer_data[i + 1] > 0:
                if pointer_flag == False:
                    one_pointer_start = i
                    pointer_flag = True
            if pointer_flag:
                if pointer_data[i] == 0 and pointer_data[i + 1] == 0:
                    one_pointer_end = i - 1
                    pointer_location = (one_pointer_start + one_pointer_end) / 2
                    one_pointer_start = 0
                    one_pointer_end = 0
                    pointer_flag = False

        scale_num = len(scale_location)
        scales = -1
        ratio = -1
        if scale_num > 0:
            for i in range(scale_num - 1):
                if scale_location[i] <= pointer_location and pointer_location < scale_location[i + 1]:
                    scales = i + (pointer_location - scale_location[i]) / (
                        scale_location[i + 1] - scale_location[i] + 1e-05) + 1
            ratio = (pointer_location - scale_location[0]) / (scale_location[scale_num - 1] - scale_location[0] + 1e-05)
        result = {'scale_num': scale_num, 'scales': scales, 'ratio': ratio}
        return result

    def predict(self,
                im_file: Union[str, np.ndarray],
                score_threshold: float = 0.5,
                seg_batch_size: int = 2,
                erode_kernel: int = 4,
                use_erode: bool = True,
                visualization: bool = False,
                save_dir: str = 'output'):

        if isinstance(im_file, str):
            im = cv2.imread(im_file).astype('float32')
        else:
            im = im_file.copy()
        det_results = self.detector.predict(im)
        filtered_results = list()
        for res in det_results:
            if res['score'] > score_threshold:
                filtered_results.append(res)

        resized_meters = list()

        for res in filtered_results:
            xmin, ymin, w, h = res['bbox']
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(im.shape[1], int(xmin + w - 1))
            ymax = min(im.shape[0], int(ymin + h - 1))
            sub_image = im[ymin:(ymax + 1), xmin:(xmax + 1), :]

            # Resize the image with shape (METER_SHAPE, METER_SHAPE)
            meter_shape = sub_image.shape
            scale_x = float(METER_SHAPE) / float(meter_shape[1])
            scale_y = float(METER_SHAPE) / float(meter_shape[0])
            meter_meter = cv2.resize(sub_image, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            meter_meter = meter_meter.astype('float32')
            resized_meters.append(meter_meter)

        meter_num = len(resized_meters)
        seg_results = list()
        for i in range(0, meter_num, seg_batch_size):
            im_size = min(meter_num, i + seg_batch_size)
            meter_images = list()
            for j in range(i, im_size):
                meter_images.append(resized_meters[j - i])

            result = self.segmenter.batch_predict(transforms=self.seg_transform, img_file_list=meter_images)

            if use_erode:
                kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
                for i in range(len(result)):
                    result[i]['label_map'] = cv2.erode(result[i]['label_map'], kernel)
            seg_results.extend(result)

        results = list()
        for i, seg_result in enumerate(seg_results):
            result = self.read_process(seg_result['label_map'])
            results.append(result)

        meter_values = list()
        for i, result in enumerate(results):
            if result['scale_num'] > TYPE_THRESHOLD:
                value = result['scales'] * METER_CONFIG[0]['scale_value']
            else:
                value = result['scales'] * METER_CONFIG[1]['scale_value']
            meter_values.append(value)
            print("-- Meter {} -- result: {} --\n".format(i, value))
        # visualize the results
        visual_results = list()
        for i, res in enumerate(filtered_results):
            # Use `score` to represent the meter value
            res['score'] = meter_values[i]
            visual_results.append(res)
        if visualization:
            pdx.det.visualize(im_file, visual_results, -1, save_dir=save_dir)

        return visual_results

    @serving
    def serving_method(self, image: str, **kwargs):
        """
        Run as a service.
        """
        images_decode = base64_to_cv2(image)
        results = self.predict(im_file=images_decode, **kwargs)
        res = []
        for result in results:
            result['category_id'] = int(result['category_id'])
            result['bbox'] = [float(value) for value in result['bbox']]
            result['category'] = str(result['category'])
            res.append(result)
        return res

    @runnable
    def run_cmd(self, argvs: list):
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
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(im_file=args.input_path)
        return results

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
