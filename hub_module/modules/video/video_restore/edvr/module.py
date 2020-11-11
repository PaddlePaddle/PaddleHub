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
import glob
import time

from tqdm import tqdm
import cv2
import numpy as np
import paddle
from PIL import Image
from paddle.utils.download import get_path_from_url
from paddlehub.module.module import moduleinfo, serving, Module

import edvr.utils as U

EDVR_WEIGHT_URL = 'https://paddlegan.bj.bcebos.com/applications/edvr_infer_model.tar'


class EDVRDataset:
    def __init__(self, frame_paths):
        self.frames = frame_paths

    def __getitem__(self, index):
        indexs = U.get_test_neighbor_frames(index, 5, len(self.frames))
        frame_list = []
        for i in indexs:
            img = U.read_img(self.frames[i])
            frame_list.append(img)

        img_LQs = np.stack(frame_list, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_LQs = np.transpose(img_LQs, (0, 3, 1, 2)).astype('float32')

        return img_LQs, self.frames[index]

    def __len__(self):
        return len(self.frames)


@moduleinfo(name="edvr",
            type="CV/image_editing",
            author="paddlepaddle",
            author_email="",
            summary="EDVR is a super resolution model",
            version="1.0.0")
class EDVRPredictor(Module):
    def _initialize(self, output_path='output', weight_path=None):
        paddle.enable_static()
        self.input = input
        self.output = os.path.join(output_path, 'EDVR')

        if weight_path is None:
            cur_path = os.path.abspath(os.path.dirname(__file__))
            weight_path = get_path_from_url(EDVR_WEIGHT_URL, cur_path)

        self.weight_path = weight_path

        self.build_inference_model()

    def build_inference_model(self):
        if paddle.in_dynamic_mode():
            # todo self.model = build_model(self.cfg)
            pass
        else:
            place = paddle.fluid.framework._current_expected_place()
            self.exe = paddle.fluid.Executor(place)
            file_names = os.listdir(self.weight_path)
            for file_name in file_names:
                if file_name.find('model') > -1:
                    model_file = file_name
                elif file_name.find('param') > -1:
                    param_file = file_name

            self.program, self.feed_names, self.fetch_targets = paddle.static.load_inference_model(
                dirname=self.weight_path,
                executor=self.exe,
                model_filename=model_file,
                params_filename=param_file)

    def base_forward(self, inputs):
        if paddle.in_dynamic_mode():
            out = self.model(inputs)
        else:
            feed_dict = {}
            if isinstance(inputs, dict):
                feed_dict = inputs
            elif isinstance(inputs, (list, tuple)):
                for i, feed_name in enumerate(self.feed_names):
                    feed_dict[feed_name] = inputs[i]
            else:
                feed_dict[self.feed_names[0]] = inputs

            out = self.exe.run(self.program,
                               fetch_list=self.fetch_targets,
                               feed=feed_dict)

        return out

    def is_image(self, input):
        try:
            img = Image.open(input)
            _ = img.size
            return True
        except:
            return False

    def predict(self, video_path):
        vid = video_path
        base_name = os.path.basename(vid).split('.')[0]
        output_path = os.path.join(self.output, base_name)
        pred_frame_path = os.path.join(output_path, 'frames_pred')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not os.path.exists(pred_frame_path):
            os.makedirs(pred_frame_path)

        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = U.video2frames(vid, output_path)

        frames = sorted(glob.glob(os.path.join(out_path, '*.png')))

        dataset = EDVRDataset(frames)

        periods = []
        cur_time = time.time()
        for infer_iter, data in enumerate(tqdm(dataset)):
            data_feed_in = [data[0]]

            outs = self.base_forward(np.array(data_feed_in))

            infer_result_list = [item for item in outs]

            frame_path = data[1]

            img_i = U.get_img(infer_result_list[0])
            U.save_img(
                img_i,
                os.path.join(pred_frame_path, os.path.basename(frame_path)))

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

        frame_pattern_combined = os.path.join(pred_frame_path, '%08d.png')
        vid_out_path = os.path.join(self.output,
                                    '{}_edvr_out.mp4'.format(base_name))
        U.frames2video(frame_pattern_combined, vid_out_path, str(int(fps)))
        print('save video result at ', vid_out_path)

        return frame_pattern_combined, vid_out_path