#coding:utf-8
import argparse
import os
import ast

import paddle.fluid as fluid
import paddlehub as hub
import numpy as np
from paddlehub.reader.cv_reader import ObjectDetectionReader
from paddlehub.dataset.base_cv_dataset import ObjectDetectionDataset

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--use_gpu",            type=ast.literal_eval,  default=False,                      help="Whether use GPU for predict.")
parser.add_argument("--checkpoint_dir",     type=str,               default="paddlehub_finetune_ckpt",  help="Path to save log data.")
parser.add_argument("--batch_size",         type=int,               default=2,                         help="Total examples' number in batch for training.")
parser.add_argument("--module",             type=str,               default="ssd",                 help="Module used as a feature extractor.")
parser.add_argument("--dataset",            type=str,               default="coco10",                  help="Dataset to finetune.")
parser.add_argument("--use_pyreader",       type=ast.literal_eval,  default=False,                      help="Whether use pyreader to feed data.")
# yapf: enable.

module_map = {
    "ssd": "ssd_mobilenet_v1_pascal",
}


def predict(args):
    module_dir = '/Users/zhaopenghao/projects/HubModule/image/object_detection/ssd/v1.0.0/ssd_mobilenet_v1_pascal.hub_module'
    version = 'v3.0.0'
    sig_name = 'feature_map'  # "multi_scale_feature"
    module = hub.Module(module_dir=[module_dir, version])
    input_dict, output_dict, program = module.context(
        trainable=True, sign_name=sig_name)

    ds = ObjectDetectionDataset(model_type='ssd')
    ds.base_path = '/Users/zhaopenghao/Downloads/coco_10'
    ds.train_image_dir = 'val'
    ds.train_list_file = 'annotations/val.json'
    ds.validate_image_dir = 'val'
    ds.validate_list_file = 'annotations/val.json'
    ds.test_image_dir = 'val'
    ds.test_list_file = 'annotations/val.json'
    # ds.num_labels = 81
    # Todo: handle ds.num_labels refresh
    print(ds.label_dict())
    print("ds.num_labels", ds.num_labels)

    data_reader = ObjectDetectionReader(1, 1, dataset=ds, model_type='ssd')

    fetch_list = [
        'module11', 'module13', 'module14', 'module15', 'module16', 'module17'
    ]
    feature_map = [output_dict[vname] for vname in fetch_list]

    img = input_dict["image"]
    feed_list = [img.name]

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
        feed_list=feed_list,
        feature=feature_map,
        num_classes=ds.num_labels,
        model_type='ssd',
        config=config)

    data = ["./test/test_img_bird.jpg", "./test/test_img_cat.jpg",]
    label_map = ds.label_dict()
    index = 0
    # get classification result
    run_states = task.predict(data=data)
    results = [run_state.run_results for run_state in run_states]
    c = 1
    for outs in results:
        # get predict index
        # Todo: handle keys dynamically
        keys = ['bbox', 'im_id']
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        # print(res['bbox'])
        # batch_num = len(res['bbox'][1][0])
        # res['im_id'] = ([[i,] for i in range(batch_num)], None)
        print(res['im_id'])
        from paddlehub.contrib.ppdet.utils.coco_eval import bbox2out
        is_bbox_normalized = False
        clsid2catid = {}
        for k in label_map:
            clsid2catid[k] = k
        bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)
        print(bbox_results)


if __name__ == "__main__":
    args = parser.parse_args()
    # if not args.module in module_map:
    #     hub.logger.error("module should in %s" % module_map.keys())
    #     exit(1)
    # args.module = module_map[args.module]

    predict(args)
