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
import base64
import argparse
from typing import List, Union, Tuple
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
from paddlehub.module.module import serving, RunModule, runnable
from paddlehub.utils.utils import base64_to_cv2, cv2_to_base64, Version


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
        labels = labels.astype('int64')

        preds, feature = self(images)

        loss, _ = F.softmax_with_cross_entropy(preds, labels, return_softmax=True, axis=1)
        loss = paddle.mean(loss)
        acc = paddle.metric.accuracy(preds, labels)
        return {'loss': loss, 'metrics': {'acc': acc}}

    def predict(self, images: List[np.ndarray], batch_size: int = 1, top_k: int = 1) -> List[dict]:
        '''
        Predict images

        Args:
            images(list[numpy.ndarray]) : Images to be predicted, consist of np.ndarray in bgr format.
            batch_size(int) : Batch size for prediciton.
            top_k(int) : Output top k result of each image.

        Returns:
            results(list[dict]) : The prediction result of each input image
        '''
        self.eval()
        with paddle.no_grad():
            res = []
            total_num = len(images)
            loop_num = int(np.ceil(total_num / batch_size))

            for iter_id in range(loop_num):
                batch_data = []
                handle_id = iter_id * batch_size
                for image_id in range(batch_size):
                    try:
                        image = self.transforms(images[handle_id + image_id])
                        batch_data.append(image)
                    except:
                        pass
                batch_image = np.array(batch_data, dtype='float32')
                preds, feature = self(paddle.to_tensor(batch_image))
                preds = F.softmax(preds, axis=1).numpy()
                pred_idxs = np.argsort(preds)[:, ::-1][:, :top_k]

                for i, pred in enumerate(pred_idxs):
                    res_dict = {}
                    for k in pred:
                        class_name = self.labels[int(k)]
                        res_dict[class_name] = preds[i][k]

                    res.append(res_dict)

            return res

    @serving
    def serving_method(self, images: list, top_k: int, **kwargs):
        """
        Run as a service.
        """
        top_k = int(top_k)
        images_decode = [base64_to_cv2(image) for image in images]
        resdicts = self.predict(images=images_decode, top_k=top_k, **kwargs)
        final = {}
        for resdict in resdicts:
            for key, value in resdict.items():
                resdict[key] = float(value)
        final['data'] = resdicts
        return final

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
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(images=[args.input_path], top_k=args.top_k)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument('--top_k', type=int, default=1, help="top_k classification result.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


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
        if Version(paddle.__version__) >= '2.1' or Version(paddle.__version__) == '0.0.0':
            img = self.preprocess(batch)
        else:
            img = self.preprocess(batch[0])

        out_class, out_reg = self(img['A'], img['hint_B'], img['mask_B'])

        # loss
        loss_ce = F.cross_entropy(out_class, img['real_B_enc'][:, :1, :, :], axis=1)
        loss_ce = paddle.mean(loss_ce)
        loss_G_L1_reg = paddle.sum(paddle.abs(img['B'] - out_reg), axis=1, keepdim=True)
        loss_G_L1_reg = paddle.mean(loss_G_L1_reg)
        loss = loss_ce + loss_G_L1_reg
        return {'loss': loss}

    def predict(self, images: list, visualization: bool = True, batch_size: int = 1, save_path: str = 'colorization'):
        '''
        Colorize images

        Args:
            images(list[str|np.ndarray]) : Images path or BGR image to be colorized.
            visualization(bool): Whether to save colorized images.
            batch_size(int): Batch size for prediciton.
            save_path(str) : Path to save colorized images.

        Returns:
            res(list[dict]) : The prediction result of each input image
        '''
        self.eval()
        with paddle.no_grad():
            lab2rgb = T.LAB2RGB()
            res = []
            total_num = len(images)
            loop_num = int(np.ceil(total_num / batch_size))
            for iter_id in range(loop_num):
                batch_data = []
                handle_id = iter_id * batch_size
                for image_id in range(batch_size):
                    try:
                        image = self.transforms(images[handle_id + image_id])
                        batch_data.append(image)
                    except:
                        pass
                batch_data = np.array(batch_data)
                im = self.preprocess(batch_data)
                out_class, out_reg = self(im['A'], im['hint_B'], im['mask_B'])

                visual_ret = OrderedDict()
                for i in range(im['A'].shape[0]):
                    gray = lab2rgb(np.concatenate((im['A'].numpy(), np.zeros(im['B'].shape)), axis=1))[i]
                    gray = np.clip(np.transpose(gray, (1, 2, 0)), 0, 1) * 255
                    visual_ret['gray'] = gray.astype(np.uint8)
                    hint = lab2rgb(np.concatenate((im['A'].numpy(), im['hint_B'].numpy()), axis=1))[i]
                    hint = np.clip(np.transpose(hint, (1, 2, 0)), 0, 1) * 255
                    visual_ret['hint'] = hint.astype(np.uint8)
                    real = lab2rgb(np.concatenate((im['A'].numpy(), im['B'].numpy()), axis=1))[i]
                    real = np.clip(np.transpose(real, (1, 2, 0)), 0, 1) * 255
                    visual_ret['real'] = real.astype(np.uint8)
                    fake = lab2rgb(np.concatenate((im['A'].numpy(), out_reg.numpy()), axis=1))[i]
                    fake = np.clip(np.transpose(fake, (1, 2, 0)), 0, 1) * 255
                    visual_ret['fake_reg'] = fake.astype(np.uint8)

                    if visualization:
                        if isinstance(images[handle_id + i], str):
                            org_img = cv2.imread(images[handle_id + i]).astype('float32')
                        else:
                            org_img = images[handle_id + i]
                        h, w, c = org_img.shape
                        fake_name = "fake_" + str(time.time()) + ".png"
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        fake_path = os.path.join(save_path, fake_name)
                        visual_gray = Image.fromarray(visual_ret['fake_reg'])
                        visual_gray = visual_gray.resize((w, h), Image.BILINEAR)
                        visual_gray.save(fake_path)

                    res.append(visual_ret)
            return res

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        visual = self.predict(images=images_decode, **kwargs)
        final = {}
        for i, visual_ret in enumerate(visual):
            h, w, c = images_decode[i].shape
            for key, value in visual_ret.items():
                value = cv2.resize(cv2.cvtColor(value, cv2.COLOR_RGB2BGR), (w, h), cv2.INTER_NEAREST)
                visual_ret[key] = cv2_to_base64(value)
        final['data'] = visual
        return final

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
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(images=[args.input_path], visualization=args.visualization, save_path=args.output_dir)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='colorization', help="save visualization result.")
        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")


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
        with paddle.no_grad():
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

    def predict(self,
                origin: list,
                style: Union[str, np.ndarray],
                batch_size: int = 1,
                visualization: bool = True,
                save_path: str = 'style_tranfer'):
        '''
        Colorize images

        Args:
            origin(list[str|np.array]): Content image path or BGR image.
            style(str|np.array): Style image path or BGR image.
            batch_size(int): Batch size for prediciton.
            visualization(bool): Whether to save colorized images.
            save_path(str) : Path to save colorized images.

        Returns:
            output(list[np.ndarray]) : The style transformed images with bgr mode.
        '''
        self.eval()
        with paddle.no_grad():
            style = paddle.to_tensor(self.transform(style).astype('float32'))
            style = style.unsqueeze(0)

            res = []
            total_num = len(origin)
            loop_num = int(np.ceil(total_num / batch_size))
            for iter_id in range(loop_num):
                batch_data = []
                handle_id = iter_id * batch_size
                for image_id in range(batch_size):
                    try:
                        image = self.transform(origin[handle_id + image_id])
                        batch_data.append(image.astype('float32'))
                    except:
                        pass

                batch_image = np.array(batch_data)
                content = paddle.to_tensor(batch_image)

                self.setTarget(style)
                output = self(content)
                for num in range(batch_size):
                    out = paddle.clip(output[num].transpose((1, 2, 0)), 0, 255).numpy().astype(np.uint8)
                    res.append(out)
                    if visualization:
                        style_name = "style_" + str(time.time()) + ".png"
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        path = os.path.join(save_path, style_name)
                        cv2.imwrite(path, out)
            return res

    @serving
    def serving_method(self, images: list, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images[0]]
        style_decode = base64_to_cv2(images[1])
        results = self.predict(origin=images_decode, style=style_decode, **kwargs)
        final = {}
        final['data'] = [cv2_to_base64(result) for result in results]
        return final

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
        self.add_module_config_arg()
        self.add_module_input_arg()
        args = self.parser.parse_args(argvs)
        results = self.predict(
            origin=[args.input_path],
            style=args.style_path,
            save_path=args.output_dir,
            visualization=args.visualization)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """

        self.arg_config_group.add_argument(
            '--output_dir', type=str, default='style_tranfer', help="The directory to save output images.")

        self.arg_config_group.add_argument(
            '--visualization', type=bool, default=True, help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument('--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument('--style_path', type=str, help="path to style image.")


class ImageSegmentationModule(ImageServing, RunModule):
    def training_step(self, batch: List[paddle.Tensor], batch_idx: int) -> dict:
        '''
        One step for training, which should be called as forward computation.

        Args:
            batch(list[paddle.Tensor]): The one batch data, which contains images, ground truth boxes, labels and scores.
            batch_idx(int): The index of batch.

        Returns:
            results(dict): The model outputs, such as loss.

        '''

        label = batch[1].astype('int64')
        criterionCE = nn.loss.CrossEntropyLoss()
        logits = self(batch[0])
        loss = 0
        for i in range(len(logits)):
            logit = logits[i]
            if logit.shape[-2:] != label.shape[-2:]:
                logit = F.interpolate(logit, label.shape[-2:], mode='bilinear')

            logit = logit.transpose([0, 2, 3, 1])
            loss_ce = criterionCE(logit, label)
            loss += loss_ce / len(logits)

        return {"loss": loss}

    def predict(self,
                images: Union[str, np.ndarray],
                batch_size: int = 1,
                visualization: bool = True,
                save_path: str = 'seg_result') -> List[np.ndarray]:
        '''
        Obtain segmentation results.

        Args:
            images(list[str|np.array]): Content image path or BGR image.
            batch_size(int): Batch size for prediciton.
            visualization(bool): Whether to save colorized images.
            save_path(str) : Path to save colorized images.

        Returns:
            output(list[np.ndarray]) : The segmentation mask.
        '''
        self.eval()
        with paddle.no_grad():
            result = []

            total_num = len(images)
            loop_num = int(np.ceil(total_num / batch_size))
            for iter_id in range(loop_num):
                batch_data = []
                handle_id = iter_id * batch_size
                for image_id in range(batch_size):
                    try:
                        image, _ = self.transform(images[handle_id + image_id])
                        batch_data.append(image)
                    except:
                        pass
                batch_image = np.array(batch_data).astype('float32')
                pred = self(paddle.to_tensor(batch_image))
                pred = paddle.argmax(pred[0], axis=1, keepdim=True, dtype='int32')

                for num in range(pred.shape[0]):
                    if isinstance(images[handle_id + num], str):
                        image = cv2.imread(images[handle_id + num])
                    else:
                        image = images[handle_id + num]
                    h, w, c = image.shape
                    pred_final = utils.reverse_transform(pred[num:num + 1], (h, w), self.transforms.transforms)
                    pred_final = paddle.squeeze(pred_final)
                    pred_final = pred_final.numpy().astype('uint8')

                    if visualization:
                        added_image = utils.visualize(images[handle_id + num], pred_final, weight=0.6)
                        pred_mask = utils.get_pseudo_color_map(pred_final)
                        pred_image_path = os.path.join(save_path, 'image', str(time.time()) + ".png")
                        pred_mask_path = os.path.join(save_path, 'mask', str(time.time()) + ".png")
                        if not os.path.exists(os.path.dirname(pred_image_path)):
                            os.makedirs(os.path.dirname(pred_image_path))
                        if not os.path.exists(os.path.dirname(pred_mask_path)):
                            os.makedirs(os.path.dirname(pred_mask_path))
                        cv2.imwrite(pred_image_path, added_image)
                        pred_mask.save(pred_mask_path)

                    result.append(pred_final)
            return result

    @serving
    def serving_method(self, images: List[str], **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        visual = self.predict(images=images_decode, **kwargs)
        final = []
        for mask in visual:
            final.append(cv2_to_base64(mask))
        return final
