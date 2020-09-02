
import time
import os

import cv2
import base64
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddlehub.module.module import serving, RunModule
import paddle.nn as nn
from collections import OrderedDict

from Colorization.process.transforms import *


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


class ImageServing(object):
    @serving
    def serving_method(self, images, **kwargs):
        """Run as a service."""
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.predict(images=images_decode, **kwargs)
        return results


class ImageClassifierModule(RunModule, ImageServing):
    def training_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        images = batch[0]
        labels = paddle.unsqueeze(batch[1], axes=-1)
        preds = self(images)
        loss, _ = fluid.layers.softmax_with_cross_entropy(preds, labels, return_softmax=True, axis=1)
        loss = fluid.layers.mean(loss)
        acc = fluid.layers.accuracy(preds, labels)
        return {'loss': loss, 'metrics': {'acc': acc}}

    def predict(self, images, top_k=1):
        images = self.transforms(images)
        if len(images.shape) == 3:
            images = images[np.newaxis, :]
        preds = self(to_variable(images))
        preds = fluid.layers.softmax(preds, axis=1).numpy()
        pred_idxs = np.argsort(preds)[::-1][:, :top_k]
        res = []
        for i, pred in enumerate(pred_idxs):
            res_dict = {}
            for k in pred:
                class_name = self.labels[int(k)]
                res_dict[class_name] = preds[i][k]
            res.append(res_dict)
        return res

    def is_better_score(self, old_score, new_score):
        return old_score['acc'] < new_score['acc']


class ImageColorizeModule(RunModule, ImageServing):

    def training_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        out_class, out_reg = self(batch[0], batch[1], batch[2])
        criterionCE = nn.loss.CrossEntropyLoss()
        loss_ce = criterionCE(out_class, batch[4][:, 0, :, :])
        loss_G_L1_reg = paddle.sum(fluid.layers.abs(batch[3] - out_reg), dim=1, keep_dim=True)
        loss_G_L1_reg = fluid.layers.mean(loss_G_L1_reg)
        loss = loss_ce + loss_G_L1_reg
        visual_ret = OrderedDict()
        psnrs=[]
        lab2rgb = ColorConvert(mode='LAB2RGB')
        process = ColorPostprocess()
        for i in range(batch[0].numpy().shape[0]):
            real = lab2rgb(np.concatenate((batch[0].numpy(), batch[3].numpy()), axis=1))[i]
            visual_ret['real'] = process(real)
            fake = lab2rgb(np.concatenate((batch[0].numpy(), out_reg.numpy()), axis=1))[i]
            visual_ret['fake_reg'] = process(fake)
            mse = np.mean((visual_ret['real'] * 1.0 - visual_ret['fake_reg'] * 1.0) ** 2)
            psnr_value = 20 * np.log10(255. / np.sqrt(mse))
            psnrs.append(psnr_value)
        psnr = to_variable(np.array(psnrs))
        return {'loss': loss, 'metrics': {'psnr': psnr}}

    def predict(self, images, visualization=True, save_path='result'):
        lab2rgb = ColorConvert(mode='LAB2RGB')
        process = ColorPostprocess()
        resize = Resize((256, 256))
        visual_ret = OrderedDict()
        im = self.transforms(images, is_train=False)
        out_class, out_reg = self(to_variable(im['A']), to_variable(im['hint_B']), to_variable(im['mask_B']))
        for i in range(im['A'].shape[0]):
            gray = lab2rgb(np.concatenate((im['A'],np.zeros(im['B'].shape)), axis=1))[i]
            visual_ret['gray'] = resize(process(gray))
            hint = lab2rgb(np.concatenate((im['A'], im['hint_B']), axis=1))[i]
            visual_ret['hint'] = resize(process(hint))
            real = lab2rgb(np.concatenate((im['A'], im['B']), axis=1))[i]
            visual_ret['real'] = resize(process(real))
            fake = lab2rgb(np.concatenate((im['A'], out_reg.numpy()), axis=1))[i]
            visual_ret['fake_reg'] = resize(process(fake))
            if visualization:
                fake_name = "fake_" + str(time.time()) + ".png"
                fake_path = os.path.join(save_path, fake_name)
                visual_gray = Image.fromarray(visual_ret['fake_reg'])
                visual_gray.save(fake_path)
            mse = np.mean((visual_ret['real'] * 1.0 - visual_ret['fake_reg'] * 1.0) ** 2)
            psnr_value = 20 * np.log10(255. / np.sqrt(mse))
            print('PSNR is :', psnr_value)
        return visual_ret






