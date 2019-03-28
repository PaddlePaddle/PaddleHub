#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path
hub_module_path=hub_module_ResNet50
cd resources
sh download.sh ResNet50
cd ..
sh create_module.sh

python retrain.py
