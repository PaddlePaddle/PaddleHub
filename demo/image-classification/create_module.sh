#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

model_name="ResNet50"

while getopts "m:" options
do
	case "$options" in
        m)
            model_name=$OPTARG;;
		?)
			echo "unknown options"
            exit 1;;
	esac
done

python create_module.py --pretrained_model=resources/${model_name}_pretrained --model ${model_name}
