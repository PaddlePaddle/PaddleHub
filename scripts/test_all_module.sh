#!/bin/bash
set -o errexit

base_path=$(cd `dirname $0`/..; pwd)
test_module_path=${base_path}/tests/modules

# install the require package
cd ${base_path}

# run all case list in the {listfile}
cd -
for test_file in `ls $test_module_path | grep test`
do
    echo "run module ${test_file}"
    python $test_module_path/$test_file
done
