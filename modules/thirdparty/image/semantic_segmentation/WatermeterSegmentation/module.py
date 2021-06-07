from __future__ import absolute_import
from __future__ import division

import os
import cv2
import argparse
import base64
import paddlex as pdx

from math import *
import time, math, re

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


'''旋转图像并剪裁'''


def rotate(
        img,  # 图片
        pt1,
        pt2,
        pt3,
        pt4,
        imgOutSrc):
    # print(pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0])**2 + (pt4[1] - pt1[1])**2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    # print("矩形的宽度",withRect, "矩形的高度", heightRect)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    # print("矩形框旋转角度", angle)

    if withRect > heightRect:
        if pt4[1] > pt1[1]:
            pass
            # print("顺时针旋转")
        else:
            # print("逆时针旋转")
            angle = -angle

    else:
        # print("逆时针旋转")
        angle = 90 - angle

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    # cv2.imwrite("imgRotation.jpg", imgRotation)

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    cv2.imwrite(imgOutSrc, imgOut)  # 裁减得到的旋转矩形框


@moduleinfo(
    name='WatermeterSegmentation',
    type='CV/semantic_segmentatio',
    author='郑博培、彭兆帅',
    author_email='2733821739@qq.com',
    summary='Digital dial segmentation of water meter',
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

    def cutPic(self, picUrl):
        # seg = hub.Module(name='WatermeterSegmentation')
        image_name = picUrl
        im = cv2.imread(image_name)
        result = self.predict(images=[im])

        # 将多边形polygon转矩形
        contours, hier = cv2.findContours(result[0]['label_map'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(type(contours[0]))
        n = 0
        m = 0
        for index, contour in enumerate(contours):
            if len(contour) > n:
                n = len(contour)
                m = index

        image = cv2.imread(image_name)
        # 获取最小的矩形
        rect = cv2.minAreaRect(contours[m])
        box = np.int0(cv2.boxPoints(rect))

        # 获取到矩形的四个点
        tmp = cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
        imgOutSrc = 'result.jpg'
        rotate(image, box[0], box[1], box[2], box[3], imgOutSrc)
        res = []
        res.append(imgOutSrc)
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
