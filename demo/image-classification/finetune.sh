#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path
hub_module_path=hub_module_ResNet50
data_dir=dataset
batch_size=32
use_gpu=False
num_epochs=20
class_dim=2
learning_rate=0.001
model_save_dir=model_save/`date +%Y%m%d%H%M%S`

while getopts "b:c:d:gh:l:n:" options
do
	case "$options" in
        b)
            batch_size=$OPTARG;;
        c)
            class_dim=$OPTARG;;
        d)
            data_dir=$OPTARG;;
        g)
            use_gpu=True;;
        h)
            hub_module_path=$OPTARG;;
        l)
            learning_rate=$OPTARG;;
        n)
            num_epochs=$OPTARG;;
        s)
            model_save_dir=$OPTARG;;
		?)
			echo "unknown options"
            exit 1;;
	esac
done

mkdir -p ${model_save_dir}

python retrain.py --batch_size=${batch_size} --class_dim=${class_dim} --data_dir=${data_dir} --use_gpu=${use_gpu} --hub_module_path ${hub_module_path} --lr ${learning_rate} --num_epochs=${num_epochs} --model_save_dir=${model_save_dir}
# nohup python retrain.py --batch_size=${batch_size} --class_dim=${class_dim} --data_dir=${data_dir} --use_gpu=${use_gpu} --hub_module_path ${hub_module_path} --lr ${learning_rate} --num_epochs=${num_epochs} --model_save_dir=${model_save_dir} > ${model_save_dir}/train.log 2>&1 &
