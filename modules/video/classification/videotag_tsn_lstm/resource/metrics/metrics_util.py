#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import io
import logging

import numpy as np

from videotag_tsn_lstm.resource.metrics.youtube8m import eval_util as youtube8m_metrics

logger = logging.getLogger(__name__)


class Metrics(object):
    def __init__(self, name, mode, metrics_args):
        """Not implemented"""
        pass

    def calculate_and_log_out(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def accumulate(self, fetch_list, info=''):
        """Not implemented"""
        pass

    def finalize_and_log_out(self, info='', savedir='./'):
        """Not implemented"""
        pass

    def reset(self):
        """Not implemented"""
        pass


class Youtube8mMetrics(Metrics):
    def __init__(self, name, mode, metrics_args):
        self.name = name
        self.mode = mode
        self.num_classes = metrics_args['MODEL']['num_classes']
        self.topk = metrics_args['MODEL']['topk']
        self.threshold = metrics_args['MODEL']['threshold']

        self.calculator = youtube8m_metrics.EvaluationMetrics(self.num_classes, self.topk)
        if self.mode == 'infer':
            self.infer_results = []

    def calculate_and_log_out(self, fetch_list, info=''):
        loss = np.mean(np.array(fetch_list[0]))
        pred = np.array(fetch_list[1])
        label = np.array(fetch_list[2])
        hit_at_one = youtube8m_metrics.calculate_hit_at_one(pred, label)
        perr = youtube8m_metrics.calculate_precision_at_equal_recall_rate(pred, label)
        gap = youtube8m_metrics.calculate_gap(pred, label)
        logger.info(info + ' , loss = {0}, Hit@1 = {1}, PERR = {2}, GAP = {3}'.format(\
                     '%.6f' % loss, '%.2f' % hit_at_one, '%.2f' % perr, '%.2f' % gap))

    def accumulate(self, fetch_list, info=''):
        if self.mode == 'infer':
            predictions = np.array(fetch_list[0])
            video_id = fetch_list[1]
            for i in range(len(predictions)):
                topk_inds = predictions[i].argsort()[0 - self.topk:]
                topk_inds = topk_inds[::-1]
                preds = predictions[i][topk_inds]
                self.infer_results.append((video_id[i], topk_inds.tolist(), preds.tolist()))
        else:
            loss = np.array(fetch_list[0])
            pred = np.array(fetch_list[1])
            label = np.array(fetch_list[2])
            self.calculator.accumulate(loss, pred, label)

    def finalize_and_log_out(self, info='', label_file='./label_3396.txt'):
        if self.mode == 'infer':
            all_res_list = []
            for index, item in enumerate(self.infer_results):
                video_id = item[0]
                f = io.open(label_file, "r", encoding="utf-8")
                fl = f.readlines()
                res = {}
                res["path"] = video_id
                res["prediction"] = {}
                for i in range(len(item[1])):
                    class_id = item[1][i]
                    class_prob = item[2][i]
                    if class_prob < self.threshold:
                        continue
                    class_name = fl[class_id].split('\n')[0]
                    res["prediction"][class_name] = class_prob
                if not res["prediction"]:
                    logger.warning("%s: No prediction exceeds the threshold = %s." % (video_id, self.threshold))
                all_res_list.append(res)
            return all_res_list
        else:
            epoch_info_dict = self.calculator.get()
            logger.info(info + '\tavg_hit_at_one: {0},\tavg_perr: {1},\tavg_loss :{2},\taps: {3},\tgap:{4}'\
                     .format(epoch_info_dict['avg_hit_at_one'], epoch_info_dict['avg_perr'], \
                             epoch_info_dict['avg_loss'], epoch_info_dict['aps'], epoch_info_dict['gap']))

    def reset(self):
        self.calculator.clear()
        if self.mode == 'infer':
            self.infer_results = []


class MetricsZoo(object):
    def __init__(self):
        self.metrics_zoo = {}

    def regist(self, name, metrics):
        assert metrics.__base__ == Metrics, "Unknow model type {}".format(type(metrics))
        self.metrics_zoo[name] = metrics

    def get(self, name, mode, cfg):
        for k, v in self.metrics_zoo.items():
            if k == name:
                return v(name, mode, cfg)
        raise KeyError(name, self.metrics_zoo.keys())


# singleton metrics_zoo
metrics_zoo = MetricsZoo()


def regist_metrics(name, metrics):
    metrics_zoo.regist(name, metrics)


def get_metrics(name, mode, cfg):
    return metrics_zoo.get(name, mode, cfg)


# sort by alphabet
regist_metrics("ATTENTIONLSTM", Youtube8mMetrics)
