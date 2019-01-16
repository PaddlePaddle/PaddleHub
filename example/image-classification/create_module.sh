#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

model_name="ResNet50"
hub_module_save_dir="./hub_module"

while getopts "m:d:" options
do
	case "$options" in
        d)
            hub_module_save_dir=$OPTARG;;
        m)
            model_name=$OPTARG;;
		?)
			echo "unknown options"
            exit 1;;
	esac
done

sh pretraind_models/download_model.sh ${model_name}
python train.py --create_module=True --pretrained_model=pretraind_models/${model_name} --model ${model_name} --use_gpu=False
