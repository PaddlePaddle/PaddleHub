#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

wget --no-check-certificate https://paddlehub.bj.bcebos.com/paddle_model/lac.tar.gz
tar xvzf lac.tar.gz
rm lac.tar.gz
