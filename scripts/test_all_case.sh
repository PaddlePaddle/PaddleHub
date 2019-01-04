#!/bin/bash
set -o errexit

base_path=$(cd `dirname $0`/..; pwd)
test_case_path=${base_path}/tests

export PYTHONPATH=$base_path:$PYTHONPATH
cd ${base_path}
pip install -r requirements.txt
cd ${test_case_path}

# only run python test case
for test_file in `ls | grep \.py`
do
	python $test_file
done
