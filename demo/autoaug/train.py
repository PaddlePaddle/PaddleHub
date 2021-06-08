# -*- coding: utf-8 -*-
#*******************************************************************************
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
#*******************************************************************************
"""

Authors: lvhaijun01@baidu.com
Date:     2020-11-26 20:57
"""
from auto_augment.autoaug.utils.yaml_config import get_config
from hub_fitter import HubFitterClassifer
import os
import argparse
import logging
import paddlehub as hub
import paddle
import paddlehub.vision.transforms as transforms
from paddlehub_utils.reader import _init_loader, PbaAugment
from paddlehub_utils.reader import _read_classes
from paddlehub_utils.trainer import CustomTrainer
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="config file",
)
parser.add_argument(
    "--workspace",
    default=None,
    help="work_space",
)
parser.add_argument(
    "--policy",
    default=None,
    help="data aug policy",
)

if __name__ == '__main__':
    args = parser.parse_args()
    config = args.config
    config = get_config(config, show=True)
    task_config = config.task_config
    data_config = config.data_config
    resource_config = config.resource_config
    algo_config = config.algo_config

    input_size = task_config.classifier.input_size
    scale_size = task_config.classifier.scale_size
    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    epochs = task_config.classifier.epochs

    policy = args.policy
    if policy is None:
        print("use normal train transform")
        TrainTransform = transforms.Compose(
            transforms=[
                transforms.Resize((input_size, input_size)),
                transforms.Permute(),
                transforms.Normalize(**normalize, channel_first=True)
            ],
            channel_first=False)
    else:
        TrainTransform = PbaAugment(
            input_size=input_size,
            scale_size=scale_size,
            normalize=normalize,
            policy=policy,
            hp_policy_epochs=epochs,
            stage="train")
    train_dataset, eval_dataset = _init_loader(config, TrainTransform=TrainTransform)
    class_to_id_dict = _read_classes(config.data_config.label_list)
    model = hub.Module(
        name=config.task_config.classifier.model_name, label_list=class_to_id_dict.keys(), load_checkpoint=None)

    optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    trainer = CustomTrainer(model=model, optimizer=optimizer, checkpoint_dir='img_classification_ckpt')
    trainer.train(train_dataset, epochs=epochs, batch_size=32, eval_dataset=eval_dataset, save_interval=10)
