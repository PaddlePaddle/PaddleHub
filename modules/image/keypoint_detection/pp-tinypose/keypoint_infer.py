# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import math
import os

import cv2
import numpy as np
import yaml

from .infer import Detector
from .keypoint_postprocess import HRNetPostProcess
from .keypoint_preprocess import expand_crop
from .visualize import visualize_pose

# Global dictionary
KEYPOINT_SUPPORT_MODELS = {'HigherHRNet': 'keypoint_bottomup', 'HRNet': 'keypoint_topdown'}


class KeyPointDetector(Detector):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        use_dark(bool): whether to use postprocess in DarkPose
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 use_dark=True):
        super(KeyPointDetector, self).__init__(
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn,
            output_dir=output_dir,
            threshold=threshold,
        )
        self.use_dark = use_dark

    def set_config(self, model_dir):
        return PredictConfig_KeyPoint(model_dir)

    def get_person_from_rect(self, image, results):
        # crop the person result from image
        valid_rects = results['boxes']
        rect_images = []
        new_rects = []
        org_rects = []
        for rect in valid_rects:
            rect_image, new_rect, org_rect = expand_crop(image, rect)
            if rect_image is None or rect_image.size == 0:
                continue
            rect_images.append(rect_image)
            new_rects.append(new_rect)
            org_rects.append(org_rect)
        return rect_images, new_rects, org_rects

    def postprocess(self, inputs, result):
        np_heatmap = result['heatmap']
        np_masks = result['masks']
        # postprocess output of predictor
        if KEYPOINT_SUPPORT_MODELS[self.pred_config.arch] == 'keypoint_bottomup':
            results = {}
            h, w = inputs['im_shape'][0]
            preds = [np_heatmap]
            if np_masks is not None:
                preds += np_masks
            preds += [h, w]
            keypoint_postprocess = HRNetPostProcess()
            kpts, scores = keypoint_postprocess(*preds)
            results['keypoint'] = kpts
            results['score'] = scores
            return results
        elif KEYPOINT_SUPPORT_MODELS[self.pred_config.arch] == 'keypoint_topdown':
            results = {}
            imshape = inputs['im_shape'][:, ::-1]
            center = np.round(imshape / 2.)
            scale = imshape / 200.
            keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
            kpts, scores = keypoint_postprocess(np_heatmap, center, scale)
            results['keypoint'] = kpts
            results['score'] = scores
            return results
        else:
            raise ValueError("Unsupported arch: {}, expect {}".format(self.pred_config.arch, KEYPOINT_SUPPORT_MODELS))

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeat number for prediction
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_heatmap, np_masks = None, None
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            heatmap_tensor = self.predictor.get_output_handle(output_names[0])
            np_heatmap = heatmap_tensor.copy_to_cpu()
            if self.pred_config.tagmap:
                masks_tensor = self.predictor.get_output_handle(output_names[1])
                heat_k = self.predictor.get_output_handle(output_names[2])
                inds_k = self.predictor.get_output_handle(output_names[3])
                np_masks = [masks_tensor.copy_to_cpu(), heat_k.copy_to_cpu(), inds_k.copy_to_cpu()]
        result = dict(heatmap=np_heatmap, masks=np_masks)
        return result

    def predict_image(self, image_list, run_benchmark=False, repeats=1, visual=True):
        results = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            inputs = self.preprocess(batch_image_list)

            # model prediction
            result = self.predict()
            # postprocess
            result = self.postprocess(inputs, result)

            if visual:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                visualize(batch_image_list, result, visual_thresh=self.threshold, save_dir=self.output_dir)

            results.append(result)
            if visual:
                print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        return results

    def predict_video(self, video_file, camera_id):
        video_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        index = 1
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            print('detect frame: %d' % (index))
            index += 1
            results = self.predict_image([frame[:, :, ::-1]], visual=False)
            im_results = {}
            im_results['keypoint'] = [results['keypoint'], results['score']]
            im = visualize_pose(frame, im_results, visual_thresh=self.threshold, returnimg=True)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of image (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.stack(imgs, axis=0).astype('float32')
    im_shape = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'])).astype('float32'))
    inputs['im_shape'] = np.stack(im_shape, axis=0)
    return inputs


class PredictConfig_KeyPoint():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in KEYPOINT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf['arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def visualize(image_list, results, visual_thresh=0.6, save_dir='output'):
    im_results = {}
    for i, image_file in enumerate(image_list):
        skeletons = results['keypoint']
        scores = results['score']
        skeleton = skeletons[i:i + 1]
        score = scores[i:i + 1]
        im_results['keypoint'] = [skeleton, score]
        visualize_pose(image_file, im_results, visual_thresh=visual_thresh, save_dir=save_dir)
