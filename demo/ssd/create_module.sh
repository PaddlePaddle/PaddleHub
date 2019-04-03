#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=hub_module_ssd

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d ssd_mobilenet_v1_pascalvoc ]
then
    sh download.sh
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
