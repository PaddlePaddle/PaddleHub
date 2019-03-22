#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)

if [ $# -ne 1 ]
then
    echo "usage: sh $0 {PRETRAINED_MODEL_NAME}"
    exit 1
fi

if [ $1 != "ResNet50" -a $1 != "ResNet101" -a $1 != "ResNet152" -a $1 != "MobileNetV2" ]
then
    echo "only suppory pretrained model in {ResNet50, ResNet101, ResNet152, MobileNetV2}"
    exit 1
fi

model_name=${1}_pretrained
model=${model_name}.zip
cd ${script_path}

if [ -d ${model_name} ]
then
    echo "model file ${model_name} is already existed"
    exit 0
fi

if [ ! -f ${model} ]
then
    wget http://paddle-imagenet-models-name.bj.bcebos.com/${model}
fi
unzip ${model}
rm ${model}
rm -rf __MACOSX
