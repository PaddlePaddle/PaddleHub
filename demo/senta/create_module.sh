#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
module_path=hub_module_senta

if [ -d $script_path/$module_path ]
then
    echo "$module_path already existed!"
    exit 0
fi

cd $script_path/resources/

if [ ! -d senta_model ]
then
    sh download.sh
fi

cd $script_path/

python create_module.py

echo "Successfully create $module_path"
