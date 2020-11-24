#!/usr/bin/env bash
export PYTHONPATH=${PYTHONPATH}:../
export PYTHONPATH=${PYTHONPATH}:/home/lvhaijun/video_recognition/utils_project/experiment_project/PaddleHub
export PYTHONPATH=${PYTHONPATH}:/home/lvhaijun/video_recognition/utils_project/experiment_project/baidu/aicc/auto-augment
echo ${PYTHONPATH}

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
config="./pba_classifier_example.yaml"
workspace="./work_dirs//autoaug_flower_mobilenetv2"
# workspace工作空间需要初始化
rm -rf ${workspace}
mkdir -p ${workspace}
CUDA_VISIBLE_DEVICES=0,1 python search.py \
    --config=${config} \
    --workspace=${workspace} #2>&1 | tee -a ${workspace}/log.txt
