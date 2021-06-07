#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import cv2
import glob

from tqdm import tqdm
import numpy as np
from PIL import Image
import paddle
import paddle.nn as nn
from paddlehub.module.module import moduleinfo, serving, Module

from realsr.rrdb import RRDBNet
import realsr.utils as U


@moduleinfo(
    name="realsr",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="realsr is a super resolution model",
    version="1.0.0")
class RealSRPredictor(Module):
    def _initialize(self, output='output', weight_path=None, load_checkpoint: str = None):
        #super(RealSRPredictor, self).__init__()
        self.input = input
        self.output = os.path.join(output, 'RealSR')
        self.model = RRDBNet(3, 3, 64, 23)

        if load_checkpoint is not None:
            state_dict = paddle.load(load_checkpoint)
            self.model.load_dict(state_dict)
            print("load custom checkpoint success")

        else:
            checkpoint = os.path.join(self.directory, 'DF2K_JPEG.pdparams')
            state_dict = paddle.load(checkpoint)
            self.model.load_dict(state_dict)
            print("load pretrained checkpoint success")

        self.model.eval()

    def norm(self, img):
        img = np.array(img).transpose([2, 0, 1]).astype('float32') / 255.0
        return img.astype('float32')

    def denorm(self, img):
        img = img.transpose((1, 2, 0))
        return (img * 255).clip(0, 255).astype('uint8')

    def run_image(self, img):
        if isinstance(img, str):
            ori_img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            # ori_img = Image.fromarray(img).convert('RGB')
            ori_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        elif isinstance(img, Image.Image):
            ori_img = img

        img = self.norm(ori_img)
        x = paddle.to_tensor(img[np.newaxis, ...])
        out = self.model(x)

        pred_img = self.denorm(out.numpy()[0])
        # pred_img = Image.fromarray(pred_img)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)

        return pred_img

    def run_video(self, video):
        base_name = os.path.basename(video).split('.')[0]
        output_path = os.path.join(self.output, base_name)
        pred_frame_path = os.path.join(output_path, 'frames_pred')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(pred_frame_path):
            os.makedirs(pred_frame_path)

        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = U.video2frames(video, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        for frame in tqdm(frames):
            pred_img = self.run_image(frame)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            pred_img = Image.fromarray(pred_img)
            frame_name = os.path.basename(frame)
            pred_img.save(os.path.join(pred_frame_path, frame_name))

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')

        vid_out_path = os.path.join(output_path, '{}_realsr_out.mp4'.format(base_name))
        U.frames2video(frame_pattern_combined, vid_out_path, str(int(fps)))
        print("save result at {}".format(vid_out_path))

        return frame_pattern_combined, vid_out_path

    def predict(self, input):
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if not U.is_image(input):
            return self.run_video(input)
        else:
            pred_img = self.run_image(input)

            out_path = None
            if self.output:
                final = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
                final = Image.fromarray(final)
                base_name = os.path.splitext(os.path.basename(input))[0]
                out_path = os.path.join(self.output, base_name + '.png')
                final.save(out_path)
                print('save result at {}'.format(out_path))

            return pred_img, out_path

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = U.base64_to_cv2(images)
        results = self.run_image(img=images_decode)
        results = U.cv2_to_base64(results)
        return results
