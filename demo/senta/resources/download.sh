#!/bin/bash
set -o nounset
set -o errexit

script_path=$(cd `dirname $0`; pwd)
cd $script_path

wget --no-check-certificate https://paddlehub.bj.bcebos.com/paddle_model/senta.tar.gz
wget --no-check-certificate https://paddlehub.bj.bcebos.com/paddle_model/train.vocab
tar xvzf senta.tar.gz
rm senta.tar.gz
