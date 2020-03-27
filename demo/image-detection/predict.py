#coding:utf-8
import argparse
import os
import ast

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np
from paddlehub.reader.cv_reader import ObjectDetectionReader
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset
from paddlehub.contrib.ppdet.utils.coco_eval import bbox2out
from paddlehub.common.detection_config import get_model_type, get_feed_list, get_mid_feature
from paddlehub.common import detection_config as dconf

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=False,                      help="Whether use GPU for predict.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt_yolov3",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=2,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="yolov3",                 help="Module used as a feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco10",                  help="Dataset to finetune.")
parser.add_argument("--use_pyreader",       type=ast.literal_eval,  default=False,                      help="Whether use pyreader to feed data.")
# yapf: enable.

module_map = {
    "yolov3": "yolov3_darknet53_coco2017",
    "ssd": "ssd_vgg16_512_coco2017",
    "faster_rcnn": "faster_rcnn_resnet50_coco2017",
}


def predict(args):
    module_name = args.module  # 'yolov3_darknet53_coco2017'
    model_type = get_model_type(module_name)  # 'yolo'
    # define data
    ds = hub.dataset.Coco10(model_type)
    print("ds.num_labels", ds.num_labels)

    data_reader = ObjectDetectionReader(1, 1, dataset=ds, model_type=model_type)

    # define model(program)
    module = hub.Module(name=module_name)
    if model_type == 'rcnn':
        input_dict, output_dict, program = module.context(trainable=True, phase='train')
        input_dict_pred, output_dict_pred, program_pred = module.context(trainable=False)
    else:
        input_dict, output_dict, program = module.context(trainable=True)
        input_dict_pred = output_dict_pred = None
    feed_list, pred_feed_list = get_feed_list(module_name, input_dict, input_dict_pred)
    feature, pred_feature = get_mid_feature(module_name, output_dict, output_dict_pred)

    config = hub.RunConfig(
        use_data_parallel=False,
        use_pyreader=args.use_pyreader,
        use_cuda=args.use_gpu,
        batch_size=args.batch_size,
        enable_memory_optim=False,
        checkpoint_dir=args.checkpoint_dir,
        strategy=hub.finetune.strategy.DefaultFinetuneStrategy())

    task = hub.DetectionTask(
        data_reader=data_reader,
        num_classes=ds.num_labels,
        feed_list=feed_list,
        feature=feature,
        predict_feed_list=pred_feed_list,
        predict_feature=pred_feature,
        model_type=model_type,
        config=config)

    data = ["./test/test_img_bird.jpg", "./test/test_img_cat.jpg",]
    label_map = ds.label_dict()
    run_states = task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    for outs in results:
        keys = ['im_shape', 'im_id', 'bbox']
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        print("im_id", res['im_id'])
        is_bbox_normalized = dconf.conf[model_type]['is_bbox_normalized']
        clsid2catid = {}
        for k in label_map:
            clsid2catid[k] = k
        bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        print(bbox_results)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.module in module_map:
        hub.logger.error("module should in %s" % module_map.keys())
        exit(1)
    args.module = module_map[args.module]

    predict(args)
