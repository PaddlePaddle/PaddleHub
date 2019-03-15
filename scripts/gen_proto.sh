#/bin/bash
protoc -I=../paddle_hub/module --python_out=../paddle_hub/module ../paddle_hub/module/module_desc.proto
protoc -I=../paddle_hub/module --python_out=../paddle_hub/module ../paddle_hub/module/check_info.proto
