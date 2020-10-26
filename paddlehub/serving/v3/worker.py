# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zmq
import os
import traceback
import sys


def run_worker(modules_info, gpu_index, addr):
    context = zmq.Context(1)
    socket = context.socket(zmq.REP)
    socket.connect(addr)
    print('Using GPU device index:', gpu_index)
    while True:
        try:
            message = socket.recv_json()
            module_name = message['module_name']
            inputs = message['inputs']
            inputs.update({'use_gpu': True})
            method = modules_info[module_name]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
            output = method(**inputs)

        except Exception as err:
            traceback.print_exc()
            output = gen_result("-1", str(err), "")
        socket.send_json(output)


if __name__ == '__main__':
    argv = sys.argv
    modules = argv[1].split(',')
    gpu_index = argv[2]
    addr = argv[3]

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    import paddlehub as hub
    from paddlehub.serving.v3.http_server import gen_result

    modules_info = {}
    for module_name in modules:
        module = hub.Module(name=module_name)
        method_name = module.serving_func_name
        serving_method = getattr(module, method_name)
        modules_info.update({module_name: serving_method})

    run_worker(modules_info, gpu_index, addr)
