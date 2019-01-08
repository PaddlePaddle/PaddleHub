#/bin/bash
protoc -I=./ --python_out=./ module_desc.proto
