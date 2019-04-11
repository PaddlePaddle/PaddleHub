#!/bin/bash
set -o nounset
set -o errexit

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

script_path=$(cd `dirname $0`; pwd)
module_path=${model_name}.hub_module

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ${model_name}_pretrained ]
then
    sh download.sh $model_name
fi

cd $script_path/

python create_module.py --pretrained_model=resources/${model_name}_pretrained --model ${model_name}

echo "Successfully create $module_path"
