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

import cv2
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image

import paddlehub.vision.transforms as T
import paddlehub.vision.functional as Func
from paddlehub.vision import utils
from paddlehub.module.module import serving, RunModule
from paddlehub.utils.utils import base64_to_cv2


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
            results(dict): The model outputs, such as loss and metrics.
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
        img = self.preprocess(batch[0])
        out_class, out_reg = self(img['A'], img['hint_B'], img['mask_B'])

        # loss
        criterionCE = nn.loss.CrossEntropyLoss()
        loss_ce = criterionCE(out_class, img['real_B_enc'][:, 0, :, :])
        loss_G_L1_reg = paddle.sum(paddle.abs(img['B'] - out_reg), axis=1, keepdim=True)
        loss_G_L1_reg = paddle.mean(loss_G_L1_reg)
        loss = loss_ce + loss_G_L1_reg

        #calculate psnr
        visual_ret = OrderedDict()
        psnrs = []
        lab2rgb = T.LAB2RGB()
        process = T.ColorPostprocess()
        for i in range(img['A'].numpy().shape[0]):
            # real = lab2rgb(np.concatenate((img['A'].numpy(), img['B'].numpy()), axis=1))[i]
            # visual_ret['real'] = process(real)
            # fake = lab2rgb(np.concatenate((img['A'].numpy(), out_reg.numpy()), axis=1))[i]
            # visual_ret['fake_reg'] = process(fake)
            # mse = np.mean((visual_ret['real'] * 1.0 - visual_ret['fake_reg'] * 1.0)**2)
            # psnr_value = 20 * np.log10(255. / np.sqrt(mse))
            psnrs.append(0)  #psnr_value)
        psnr = paddle.to_tensor(np.array(psnrs))
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
        self.eval()
        lab2rgb = T.LAB2RGB()
        process = T.ColorPostprocess()
        resize = T.Resize((256, 256))

        im = self.transforms(images, is_train=False)
        im = im[np.newaxis, :, :, :]
        im = self.preprocess(im)
        out_class, out_reg = self(im['A'], im['hint_B'], im['mask_B'])

        result = []
        visual_ret = OrderedDict()
        for i in range(im['A'].shape[0]):
            gray = lab2rgb(np.concatenate((im['A'].numpy(), np.zeros(im['B'].shape)), axis=1))[i]
            visual_ret['gray'] = resize(process(gray))
            hint = lab2rgb(np.concatenate((im['A'].numpy(), im['hint_B'].numpy()), axis=1))[i]
            visual_ret['hint'] = resize(process(hint))
            real = lab2rgb(np.concatenate((im['A'].numpy(), im['B'].numpy()), axis=1))[i]
            visual_ret['real'] = resize(process(real))
            fake = lab2rgb(np.concatenate((im['A'].numpy(), out_reg.numpy()), axis=1))[i]
            visual_ret['fake_reg'] = resize(process(fake))

            if visualization:
                img = Image.open(images)
                w, h = img.size[0], img.size[1]
                fake_name = "fake_" + str(time.time()) + ".png"
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                fake_path = os.path.join(save_path, fake_name)
                visual_gray = Image.fromarray(visual_ret['fake_reg'])
                visual_gray = visual_gray.resize((w, h), Image.BILINEAR)
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
        img = batch[0].astype('float32')
        gtbox = batch[1].astype('float32')
        gtlabel = batch[2].astype('int32')
        gtscore = batch[3].astype("float32")
        losses = []
        outputs = self(img)
        self.downsample = 32

        for i, out in enumerate(outputs):
            anchor_mask = self.anchor_masks[i]
            loss = F.yolov3_loss(
                x=out,
                gt_box=gtbox,
                gt_label=gtlabel,
                gt_score=gtscore,
                anchors=self.anchors,
                anchor_mask=anchor_mask,
                class_num=self.class_num,
                ignore_thresh=self.ignore_thresh,
                downsample_ratio=32,
                use_label_smooth=False)
            losses.append(paddle.mean(loss))
            self.downsample //= 2

        return {'loss': sum(losses)}

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
        self.eval()
        boxes = []
        scores = []
        self.downsample = 32
        im = self.transform(imgpath)
        h, w, c = utils.img_shape(imgpath)
        im_shape = paddle.to_tensor(np.array([[h, w]]).astype('int32'))
        label_names = utils.get_label_infos(filelist)
        img_data = paddle.to_tensor(np.array([im]).astype('float32'))

        outputs = self(img_data)

        for i, out in enumerate(outputs):
            anchor_mask = self.anchor_masks[i]
            mask_anchors = []
            for m in anchor_mask:
                mask_anchors.append((self.anchors[2 * m]))
                mask_anchors.append(self.anchors[2 * m + 1])

            box, score = F.yolo_box(
                x=out,
                img_size=im_shape,
                anchors=mask_anchors,
                class_num=self.class_num,
                conf_thresh=self.valid_thresh,
                downsample_ratio=self.downsample,
                name="yolo_box" + str(i))

            boxes.append(box)
            scores.append(paddle.transpose(score, perm=[0, 2, 1]))
            self.downsample //= 2

        yolo_boxes = paddle.concat(boxes, axis=1)
        yolo_scores = paddle.concat(scores, axis=2)

        pred = F.multiclass_nms(
            bboxes=yolo_boxes,
            scores=yolo_scores,
            score_threshold=self.valid_thresh,
            nms_top_k=self.nms_topk,
            keep_top_k=self.nms_posk,
            nms_threshold=self.nms_thresh,
            background_label=-1)

        bboxes = pred.numpy()
        labels = bboxes[:, 0].astype('int32')
        scores = bboxes[:, 1].astype('float32')
        boxes = bboxes[:, 2:].astype('float32')

        if visualization:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            utils.draw_boxes_on_image(imgpath, boxes, scores, labels, label_names, 0.5, save_path)

        return boxes, scores, labels


class StyleTransferModule(RunModule, ImageServing):
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
        mse_loss = nn.MSELoss()
        N, C, H, W = batch[0].shape
        batch[1] = batch[1][0].unsqueeze(0)
        self.setTarget(batch[1])

        y = self(batch[0])
        xc = paddle.to_tensor(batch[0].numpy().copy())
        y = utils.subtract_imagenet_mean_batch(y)
        xc = utils.subtract_imagenet_mean_batch(xc)
        features_y = self.getFeature(y)
        features_xc = self.getFeature(xc)
        f_xc_c = paddle.to_tensor(features_xc[1].numpy(), stop_gradient=True)
        content_loss = mse_loss(features_y[1], f_xc_c)

        batch[1] = utils.subtract_imagenet_mean_batch(batch[1])
        features_style = self.getFeature(batch[1])
        gram_style = [utils.gram_matrix(y) for y in features_style]
        style_loss = 0.
        for m in range(len(features_y)):
            gram_y = utils.gram_matrix(features_y[m])
            gram_s = paddle.to_tensor(np.tile(gram_style[m].numpy(), (N, 1, 1, 1)))
            style_loss += mse_loss(gram_y, gram_s[:N, :, :])

        loss = content_loss + style_loss

        return {'loss': loss, 'metrics': {'content gap': content_loss, 'style gap': style_loss}}

    def predict(self, origin_path: str, style_path: str, visualization: bool = True, save_path: str = 'result'):
        '''
        Colorize images

        Args:
            origin_path(str): Content image path .
            style_path(str): Style image path.
            visualization(bool): Whether to save colorized images.
            save_path(str) : Path to save colorized images.

        Returns:
            output(np.ndarray) : The style transformed images with bgr mode.
        '''
        self.eval()
        content = paddle.to_tensor(self.transform(origin_path))
        style = paddle.to_tensor(self.transform(style_path))
        content = content.unsqueeze(0)
        style = style.unsqueeze(0)

        self.setTarget(style)
        output = self(content)
        output = paddle.clip(output[0].transpose((1, 2, 0)), 0, 255).numpy()

        if visualization:
            output = output.astype(np.uint8)
            style_name = "style_" + str(time.time()) + ".png"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            path = os.path.join(save_path, style_name)
            cv2.imwrite(path, output)
        return output
