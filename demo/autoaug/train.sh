#!/usr/bin/env bash
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
config="./pba_classifier_example.yaml"
workspace="./work_dirs//autoaug_flower_mobilenetv2"
# workspace工作空间需要初始化
mkdir -p ${workspace}
policy=./work_dirs//autoaug_flower_mobilenetv2/auto_aug_config.json
CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --config=${config} \
    --policy=${policy} \
    --workspace=${workspace} 2>&1 | tee -a ${workspace}/log.txt
