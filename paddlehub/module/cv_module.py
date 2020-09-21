# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
from typing import List
from collections import OrderedDict

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

from paddlehub.module.module import serving, RunModule
from paddlehub.utils.utils import base64_to_cv2
from paddlehub.process.transforms import ConvertColorSpace, ColorPostprocess, Resize, BoxTool


class ImageServing(object):
    @serving
    def serving_method(self, images: List[str], **kwargs) -> List[dict]:
        """Run as a service."""
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images=images_decode, **kwargs)
        return results


class ImageClassifierModule(RunModule, ImageServing):
    def training_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for training, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]) : The one batch data, which contains images and labels.
            batch_idx(int) : The index of batch.

        Returns:
            results(dict) : The model outputs, such as loss and metrics.
        '''
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for validation, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]) : The one batch data, which contains images and labels.
            batch_idx(int) : The index of batch.

        Returns:
            results(dict) : The model outputs, such as metrics.
        '''
        images = batch[0]
        labels = paddle.unsqueeze(batch[1], axis=-1)

        preds = self(images)
        loss, _ = F.softmax_with_cross_entropy(preds, labels, return_softmax=True, axis=1)
        loss = paddle.mean(loss)
        acc = paddle.metric.accuracy(preds, labels)
        return {'loss': loss, 'metrics': {'acc': acc}}

    def predict(self, images: List[np.ndarray], top_k: int = 1) -> List[dict]:
        '''
        Predict images

        Args:
            images(list[numpy.ndarray]) : Images to be predicted, consist of np.ndarray in bgr format.
            top_k(int) : Output top k result of each image.

        Returns:
            results(list[dict]) : The prediction result of each input image
        '''
        images = self.transforms(images)
        if len(images.shape) == 3:
            images = images[np.newaxis, :]
        preds = self(paddle.to_tensor(images))
        preds = F.softmax(preds, axis=1).numpy()
        pred_idxs = np.argsort(preds)[::-1][:, :top_k]
        res = []
        for i, pred in enumerate(pred_idxs):
            res_dict = {}
            for k in pred:
                class_name = self.labels[int(k)]
                res_dict[class_name] = preds[i][k]
            res.append(res_dict)
        return res


class ImageColorizeModule(RunModule, ImageServing):
    def training_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for training, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]): The one batch data, which contains images and labels.
            batch_idx(int): The index of batch.

        Returns:
            results(dict) : The model outputs, such as loss and metrics.
        '''
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for validation, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]): The one batch data, which contains images and labels.
            batch_idx(int): The index of batch.

        Returns:
            results(dict) : The model outputs, such as metrics.
        '''
        out_class, out_reg = self(batch[0], batch[1], batch[2])

        criterionCE = nn.loss.CrossEntropyLoss()
        loss_ce = criterionCE(out_class, batch[4][:, 0, :, :])
        loss_G_L1_reg = paddle.sum(paddle.abs(batch[3] - out_reg), axis=1, keepdim=True)
        loss_G_L1_reg = paddle.mean(loss_G_L1_reg)
        loss = loss_ce + loss_G_L1_reg

        visual_ret = OrderedDict()
        psnrs = []
        lab2rgb = ConvertColorSpace(mode='LAB2RGB')
        process = ColorPostprocess()
        for i in range(batch[0].numpy().shape[0]):
            real = lab2rgb(np.concatenate((batch[0].numpy(), batch[3].numpy()), axis=1))[i]
            visual_ret['real'] = process(real)
            fake = lab2rgb(np.concatenate((batch[0].numpy(), out_reg.numpy()), axis=1))[i]
            visual_ret['fake_reg'] = process(fake)
            mse = np.mean((visual_ret['real'] * 1.0 - visual_ret['fake_reg'] * 1.0)**2)
            psnr_value = 20 * np.log10(255. / np.sqrt(mse))
            psnrs.append(psnr_value)
        psnr = paddle.to_variable(np.array(psnrs))
        return {'loss': loss, 'metrics': {'psnr': psnr}}

    def predict(self, images: str, visualization: bool = True, save_path: str = 'result'):
        '''
        Colorize images

        Args:
            images(str) : Images path to be colorized.
            visualization(bool): Whether to save colorized images.
            save_path(str) : Path to save colorized images.

        Returns:
            results(list[dict]) : The prediction result of each input image
        '''
        lab2rgb = ConvertColorSpace(mode='LAB2RGB')
        process = ColorPostprocess()
        resize = Resize((256, 256))
        visual_ret = OrderedDict()
        im = self.transforms(images, is_train=False)
        out_class, out_reg = self(paddle.to_tensor(im['A']), paddle.to_variable(im['hint_B']),
                                  paddle.to_variable(im['mask_B']))
        result = []

        for i in range(im['A'].shape[0]):
            gray = lab2rgb(np.concatenate((im['A'], np.zeros(im['B'].shape)), axis=1))[i]
            visual_ret['gray'] = resize(process(gray))
            hint = lab2rgb(np.concatenate((im['A'], im['hint_B']), axis=1))[i]
            visual_ret['hint'] = resize(process(hint))
            real = lab2rgb(np.concatenate((im['A'], im['B']), axis=1))[i]
            visual_ret['real'] = resize(process(real))
            fake = lab2rgb(np.concatenate((im['A'], out_reg.numpy()), axis=1))[i]
            visual_ret['fake_reg'] = resize(process(fake))

            if visualization:
                fake_name = "fake_" + str(time.time()) + ".png"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                fake_path = os.path.join(save_path, fake_name)
                visual_gray = Image.fromarray(visual_ret['fake_reg'])
                visual_gray.save(fake_path)

            mse = np.mean((visual_ret['real'] * 1.0 - visual_ret['fake_reg'] * 1.0)**2)
            psnr_value = 20 * np.log10(255. / np.sqrt(mse))
            result.append(visual_ret)
        return result


class Yolov3Module(RunModule, ImageServing):
    def training_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for training, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]): The one batch data, which contains images, ground truth boxes, labels and scores.
            batch_idx(int): The index of batch.

        Returns:
            results(dict): The model outputs, such as loss.
        '''

        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch: int, batch_idx: int) -> dict:
        '''
        One step for validation, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]): The one batch data, which contains images, ground truth boxes, labels and scores.
            batch_idx(int): The index of batch.

        Returns:
            results(dict) : The model outputs, such as metrics.
        '''
        ious = []
        boxtool = BoxTool()
        img = batch[0].astype('float32')
        B, C, W, H = img.shape
        im_shape = np.array([(W, H)] * B).astype('int32')
        im_shape = paddle.to_tensor(im_shape)

        gt_box = batch[1].astype('float32')
        gt_label = batch[2].astype('int32')
        gt_score = batch[3].astype("float32")
        loss, pred = self(img, gt_box, gt_label, gt_score, im_shape)

        for i in range(len(pred)):
            bboxes = pred[i].numpy()
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')
            iou = []

            for j, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = box
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                bbox = [x1, y1, w, h]
                bbox = np.expand_dims(boxtool.coco_anno_box_to_center_relative(bbox, H, W), 0)
                gt = gt_box[i].numpy()
                iou.append(max(boxtool.box_iou_xywh(bbox, gt)))

            ious.append(max(iou))
        ious = paddle.to_tensor(np.array(ious))

        return {'loss': loss, 'metrics': {'iou': ious}}

    def predict(self, imgpath: str, filelist: str, visualization: bool = True, save_path: str = 'result'):
        '''
        Detect images

        Args:
            imgpath(str): Image path .
            filelist(str): Path to get label name.
            visualization(bool): Whether to save result image.
            save_path(str) : Path to save detected images.

        Returns:
            boxes(np.ndarray): Predict box information.
            scores(np.ndarray): Predict score.
            labels(np.ndarray): Predict labels.
        '''
        boxtool = BoxTool()
        img = {}
        img['image'] = imgpath
        img['id'] = 0
        im, im_id, im_shape = self.transform(img, 416)
        label_names = self.get_label_infos(filelist)
        img_data = np.array([im]).astype('float32')
        img_data = paddle.to_tensor(img_data)
        im_shape = np.array([im_shape]).astype('int32')
        im_shape = paddle.to_tensor(im_shape)

        output, pred = self(img_data, None, None, None, im_shape)

        for i in range(len(pred)):
            bboxes = pred[i].numpy()
            labels = bboxes[:, 0].astype('int32')
            scores = bboxes[:, 1].astype('float32')
            boxes = bboxes[:, 2:].astype('float32')

            if visualization:
                boxtool.draw_boxes_on_image(imgpath, boxes, scores, labels, label_names, 0.5)

        return boxes, scores, labels
