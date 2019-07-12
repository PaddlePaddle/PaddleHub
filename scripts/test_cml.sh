#!/bin/bash
set -o errexit
base_path=$(cd `dirname $0`; pwd)
cd $base_path

# test install command
hub install lac
