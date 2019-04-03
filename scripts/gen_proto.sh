#/bin/bash
protoc -I=../paddlehub/module --python_out=../paddlehub/module ../paddlehub/module/module_desc.proto
protoc -I=../paddlehub/module --python_out=../paddlehub/module ../paddlehub/module/check_info.proto
protoc -I=../paddlehub/finetune --python_out=../paddlehub/finetune ../paddlehub/finetune/checkpoint.proto
