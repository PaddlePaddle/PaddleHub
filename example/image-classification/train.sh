#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

model_name=ResNet50
batch_size=32
data_dir=./dataset
class_dim=2
use_gpu=False

while getopts "m:b:c:d:g" options
do
	case "$options" in
        b)
            batch_size=$OPTARG;;
        c)
            class_dim=$OPTARG;;
        d)
            data_dir=$OPTARG;;
		m)
			model_name=$OPTARG;;
        g)
            use_gpu=True;;
		?)
			echo "unknown options"
            exit 1;;
	esac
done

python train.py --data_dir=${data_dir} --batch_size=${batch_size} --class_dim=${class_dim} --image_shape=3,224,224 --model_save_dir=output/ --lr_strategy=piecewise_decay --lr=0.1 --model=${model_name} --use_gpu=${use_gpu}
