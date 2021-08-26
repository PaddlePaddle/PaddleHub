# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import glob
import paddle
import numpy as np

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.modeling.mot.utils import Detection, get_crops, scale_coords, clip_box
from ppdet.modeling.mot.utils import Timer, load_det_results
from ppdet.modeling.mot import visualization as mot_vis

from ppdet.metrics import Metric, MOTMetric, KITTIMOTMetric
import ppdet.utils.stats as stats

from ppdet.engine.callbacks import Callback, ComposeCallback

from ppdet.utils.logger import setup_logger
from .dataset import MOTVideoStream
logger = setup_logger(__name__)

__all__ = ['Tracker']


class StreamTracker(object):
    def __init__(self, cfg, mode='eval'):
        self.cfg = cfg
        assert mode.lower() in ['test', 'eval'], \
                "mode should be 'test' or 'eval'"
        self.mode = mode.lower()
        self.optimizer = None        

        # build model
        self.model = create(cfg.architecture)

        self.status = {}
        self.start_epoch = 0


    def load_weights_jde(self, weights):
        load_weight(self.model, weights, self.optimizer)

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights, self.optimizer)

    def _eval_seq_jde(self,
                      dataloader,
                      save_dir=None,
                      show_image=False,
                      frame_rate=30,
                      draw_threshold=0):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        tracker = self.model.tracker
        tracker.max_time_lost = int(frame_rate / 30.0 * tracker.track_buffer)

        timer = Timer()
        results = []
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        for step_id, data in enumerate(dataloader):
            #print('data', data)
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            # forward
            timer.tic()
            pred_dets, pred_embs = self.model(data)
            online_targets = self.model.tracker.update(pred_dets, pred_embs)

            online_tlwhs, online_ids = [], []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tscore = t.score
                if tscore < draw_threshold: continue
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > tracker.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(tscore)
            timer.toc()

            # save results
            results.append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            self.save_results(data, frame_id, online_ids, online_tlwhs,
                              online_scores, timer.average_time, show_image,
                              save_dir)
            frame_id += 1

        return results, frame_id, timer.average_time, timer.calls

    def _eval_seq_sde(self,
                      dataloader,
                      save_dir=None,
                      show_image=False,
                      frame_rate=30,
                      det_file='',
                      draw_threshold=0):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        tracker = self.model.tracker
        use_detector = False if not self.model.detector else True

        timer = Timer()
        results = []
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        self.model.reid.eval()
        if not use_detector:
            dets_list = load_det_results(det_file, len(dataloader))
            logger.info('Finish loading detection results file {}.'.format(
                det_file))

        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            ori_image = data['ori_image']
            input_shape = data['image'].shape[2:]
            im_shape = data['im_shape']
            scale_factor = data['scale_factor']
            timer.tic()
            if not use_detector:
                dets = dets_list[frame_id]
                bbox_tlwh = paddle.to_tensor(dets['bbox'], dtype='float32')
                pred_scores = paddle.to_tensor(dets['score'], dtype='float32')
                if pred_scores < draw_threshold: continue
                if bbox_tlwh.shape[0] > 0:
                    pred_bboxes = paddle.concat(
                        (bbox_tlwh[:, 0:2],
                         bbox_tlwh[:, 2:4] + bbox_tlwh[:, 0:2]),
                        axis=1)
                else:
                    pred_bboxes = []
                    pred_scores = []
            else:
                outs = self.model.detector(data)
                if outs['bbox_num'] > 0:
                    pred_bboxes = scale_coords(outs['bbox'][:, 2:], input_shape,
                                               im_shape, scale_factor)
                    pred_scores = outs['bbox'][:, 1:2]
                else:
                    pred_bboxes = []
                    pred_scores = []

            pred_bboxes = clip_box(pred_bboxes, input_shape, im_shape,
                                   scale_factor)
            bbox_tlwh = paddle.concat(
                (pred_bboxes[:, 0:2],
                 pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1),
                axis=1)

            crops, pred_scores = get_crops(
                pred_bboxes, ori_image, pred_scores, w=64, h=192)
            crops = paddle.to_tensor(crops)
            pred_scores = paddle.to_tensor(pred_scores)

            data.update({'crops': crops})
            features = self.model(data)
            features = features.numpy()
            detections = [
                Detection(tlwh, score, feat)
                for tlwh, score, feat in zip(bbox_tlwh, pred_scores, features)
            ]
            self.model.tracker.predict()
            online_targets = self.model.tracker.update(detections)

            online_tlwhs = []
            online_scores = []
            online_ids = []
            for track in online_targets:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                online_tlwhs.append(track.to_tlwh())
                online_scores.append(1.0)
                online_ids.append(track.track_id)
            timer.toc()

            # save results
            results.append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            self.save_results(data, frame_id, online_ids, online_tlwhs,
                              online_scores, timer.average_time, show_image,
                              save_dir)
            frame_id += 1

        return results, frame_id, timer.average_time, timer.calls


    def mot_predict(self,
                    video_stream,
                    output_dir,
                    data_type='mot',
                    model_type='JDE',
                    visualization=True,
                    draw_threshold=0.5,
                    from_device=False):
        assert video_stream is not None, \
            "--video_file or --image_dir should be set."

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_root = os.path.join(output_dir, 'mot_results')
        if not os.path.exists(result_root): os.makedirs(result_root)
        assert data_type in ['mot', 'kitti'], \
            "data_type should be 'mot' or 'kitti'"
        assert model_type in ['JDE', 'DeepSORT', 'FairMOT'], \
            "model_type should be 'JDE', 'DeepSORT' or 'FairMOT'"
        seq = os.path.splitext(os.path.basename(video_stream))[0]
        self.dataset = MOTVideoStream(video_stream, from_device=from_device,keep_ori_im=True)

        save_dir = os.path.join(output_dir, 'mot_outputs',
                                seq) if visualization else None

        dataloader = create('MOTVideoStreamReader')(self.dataset, 0)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))

        with paddle.no_grad():
            if model_type in ['JDE', 'FairMOT']:
                results, nf, ta, tc = self._eval_seq_jde(
                    dataloader,
                    save_dir=save_dir,
                    draw_threshold=draw_threshold)
            elif model_type in ['DeepSORT']:
                results, nf, ta, tc = self._eval_seq_sde(
                    dataloader,
                    save_dir=save_dir,
                    det_file=os.path.join(det_results_dir,
                                          '{}.txt'.format(seq)),
                    draw_threshold=draw_threshold)
            else:
                raise ValueError(model_type)

        self.write_mot_results(result_filename, results, data_type)

        if visualization:
            output_video_path = os.path.join(save_dir, '..',
                                             '{}_vis.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {}'.format(
                save_dir, output_video_path)
            os.system(cmd_str)
            logger.info('Save video in {}'.format(output_video_path))

    def write_mot_results(self, filename, results, data_type='mot'):
        if data_type in ['mot', 'mcmot', 'lab']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{score},-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} car 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, tscores, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(
                        frame=frame_id,
                        id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        w=w,
                        h=h,
                        score=score)
                    f.write(line)
        logger.info('MOT results save in {}'.format(filename))

    def save_results(self, data, frame_id, online_ids, online_tlwhs,
                     online_scores, average_time, show_image, save_dir):
        if show_image or save_dir is not None:
            assert 'ori_image' in data
            img0 = data['ori_image'].numpy()[0]
            online_im = mot_vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=1. / average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)),
                online_im)
