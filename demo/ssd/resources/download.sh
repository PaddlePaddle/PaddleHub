#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

wget --no-check-certificate https://paddlehub.bj.bcebos.com/paddle_model/ssd_mobilenet_v1_pascalvoc.tar.gz
tar xvzf ssd_mobilenet_v1_pascalvoc.tar.gz
rm ssd_mobilenet_v1_pascalvoc.tar.gz
