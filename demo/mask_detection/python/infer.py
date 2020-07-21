# coding: utf8
# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import ast
import time
import json
import argparse

import numpy as np
import cv2

import paddle.fluid as fluid

from PIL import Image
from PIL import ImageDraw

import argparse


def parse_args():
    parser = argparse.ArgumentParser('mask detection.')
    parser.add_argument(
        '--models_dir', type=str, default='', help='path of models.')
    parser.add_argument(
        '--img_paths', type=str, default='', help='path of images')
    parser.add_argument(
        '--video_path', type=str, default='', help='path of video.')
    parser.add_argument(
        '--use_camera',
        type=bool,
        default=False,
        help='switch detect video or camera, default:video.')
    parser.add_argument(
        '--open_imshow',
        type=bool,
        default=False,
        help='visualize video detection results in real time.')
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=False,
        help='switch cpu/gpu, default:cpu.')
    args = parser.parse_args()
    return args


class FaceResult:
    def __init__(self, rect_data, rect_info):
        self.rect_info = rect_info
        self.rect_data = rect_data
        self.class_id = -1
        self.score = 0.0


def VisualizeResult(im, faces):
    LABELS = ['NO_MASK', 'MASK']
    COLORS = [(0, 0, 255), (0, 255, 0)]
    for face in faces:
        label = LABELS[face.class_id]
        color = COLORS[face.class_id]
        left, right, top, bottom = [int(item) for item in face.rect_info]
        label_position = (left, top)
        cv2.putText(im, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2, cv2.LINE_AA)
        cv2.rectangle(im, (left, top), (right, bottom), color, 3)
    return im


def LoadModel(model_dir, use_gpu=False):
    config = fluid.core.AnalysisConfig(model_dir + '/__model__',
                                       model_dir + '/__params__')
    if use_gpu:
        config.enable_use_gpu(100, 0)
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
    config.disable_glog_info()
    config.switch_specify_input_names(True)
    config.enable_memory_optim()
    return fluid.core.create_paddle_predictor(config)


class MaskClassifier:
    def __init__(self, model_dir, mean, scale, use_gpu=False):
        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.predictor = LoadModel(model_dir, use_gpu)
        self.EVAL_SIZE = (128, 128)

    def Preprocess(self, faces):
        h, w = self.EVAL_SIZE[1], self.EVAL_SIZE[0]
        inputs = []
        for face in faces:
            im = cv2.resize(
                face.rect_data, (128, 128),
                fx=0,
                fy=0,
                interpolation=cv2.INTER_CUBIC)
            # HWC -> CHW
            im = im.swapaxes(1, 2)
            im = im.swapaxes(0, 1)
            # Convert to float
            im = im[:, :, :].astype('float32') / 256.0
            # im  = (im - mean) * scale
            im = im - self.mean
            im = im * self.scale
            im = im[np.newaxis, :, :, :]
            inputs.append(im)
        return inputs

    def Postprocess(self, output_data, faces):
        argmx = np.argmax(output_data, axis=1)
        for idx in range(len(faces)):
            faces[idx].class_id = argmx[idx]
            faces[idx].score = output_data[idx][argmx[idx]]
        return faces

    def Predict(self, faces):
        inputs = self.Preprocess(faces)
        if len(inputs) != 0:
            input_data = np.concatenate(inputs)
            im_tensor = fluid.core.PaddleTensor(
                input_data.copy().astype('float32'))
            output_data = self.predictor.run([im_tensor])[0]
            output_data = output_data.as_ndarray()
            self.Postprocess(output_data, faces)


class FaceDetector:
    def __init__(self, model_dir, mean, scale, use_gpu=False, threshold=0.7):
        self.mean = np.array(mean).reshape((3, 1, 1))
        self.scale = np.array(scale).reshape((3, 1, 1))
        self.threshold = threshold
        self.predictor = LoadModel(model_dir, use_gpu)

    def Preprocess(self, image, shrink):
        h, w = int(image.shape[1] * shrink), int(image.shape[0] * shrink)
        im = cv2.resize(
            image, (h, w), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # HWC -> CHW
        im = im.swapaxes(1, 2)
        im = im.swapaxes(0, 1)
        # Convert to float
        im = im[:, :, :].astype('float32')
        # im  = (im - mean) * scale
        im = im - self.mean
        im = im * self.scale
        im = im[np.newaxis, :, :, :]
        return im

    def Postprocess(self, output_data, ori_im, shrink):
        det_out = []
        h, w = ori_im.shape[0], ori_im.shape[1]
        for out in output_data:
            class_id = int(out[0])
            score = out[1]
            xmin = (out[2] * w)
            ymin = (out[3] * h)
            xmax = (out[4] * w)
            ymax = (out[5] * h)
            wd = xmax - xmin
            hd = ymax - ymin
            valid = (xmax >= xmin and xmin > 0 and ymax >= ymin and ymin > 0)
            if score > self.threshold and valid:
                roi_rect = ori_im[int(ymin):int(ymax), int(xmin):int(xmax)]
                det_out.append(FaceResult(roi_rect, [xmin, xmax, ymin, ymax]))
        return det_out

    def Predict(self, image, shrink):
        ori_im = image.copy()
        im = self.Preprocess(image, shrink)
        im_tensor = fluid.core.PaddleTensor(im.copy().astype('float32'))
        output_data = self.predictor.run([im_tensor])[0]
        output_data = output_data.as_ndarray()
        return self.Postprocess(output_data, ori_im, shrink)


def predict_images(args):
    detector = FaceDetector(
        model_dir=args.models_dir + '/pyramidbox_lite/',
        mean=[104.0, 177.0, 123.0],
        scale=[0.007843, 0.007843, 0.007843],
        use_gpu=args.use_gpu,
        threshold=0.7)

    classifier = MaskClassifier(
        model_dir=args.models_dir + '/mask_detector/',
        mean=[0.5, 0.5, 0.5],
        scale=[1.0, 1.0, 1.0],
        use_gpu=args.use_gpu)
    names = []
    image_paths = []
    for name in os.listdir(args.img_paths):
        if name.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            names.append(name)
            image_paths.append(os.path.join(args.img_paths, name))
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    for idx in range(len(images)):
        im = images[idx]
        det_out = detector.Predict(im, shrink=0.7)
        classifier.Predict(det_out)
        img = VisualizeResult(im, det_out)
        cv2.imwrite(os.path.join(path, names[idx] + '.result.jpg'), img)


def predict_video(args, im_shape=(1920, 1080), use_camera=False):
    if args.use_camera:
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(args.video_path)
    detector = FaceDetector(
        model_dir=args.models_dir + '/pyramidbox_lite/',
        mean=[104.0, 177.0, 123.0],
        scale=[0.007843, 0.007843, 0.007843],
        use_gpu=args.use_gpu,
        threshold=0.7)

    classifier = MaskClassifier(
        model_dir=args.models_dir + '/mask_detector/',
        mean=[0.5, 0.5, 0.5],
        scale=[1.0, 1.0, 1.0],
        use_gpu=args.use_gpu)

    path = './result'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    fps = 30
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        os.path.join(path, 'result.mp4'), fourcc, fps, (width, height))
    import time
    start_time = time.time()
    index = 0
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1
        det_out = detector.Predict(frame, shrink=0.5)
        classifier.Predict(det_out)
        end_pre = time.time()
        im = VisualizeResult(frame, det_out)
        writer.write(im)
        if args.open_imshow:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    end_time = time.time()
    print("Average prediction time per frame:", (end_time - start_time) / index)
    writer.release()


if __name__ == "__main__":
    args = parse_args()
    print(args.models_dir)
    if args.img_paths != '':
        predict_images(args)
    elif args.video_path != '' or args.use_camera:
        predict_video(args)
