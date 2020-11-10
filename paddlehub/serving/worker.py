# coding:utf-8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
import os
import json
import traceback
import sys


def run_worker(modules_info: dict, gpu_index: int, addr: str):
    '''
    Start zmq.REP as backend on specified GPU.

    Args:
        modules_info(dict): module name to serving method
        gpu_index(int): GPU device index to use
        addr(str): address of zmq.REP

    Examples:
        .. code-block:: python

            modules_info = {'lac': lexical_analise}
            run_worker(modules_info=modules_info,
                       gpu_index=0,
                       addr='ipc://backend.ipc')

    '''
    context = zmq.Context(1)
    socket = context.socket(zmq.REP)
    socket.connect(addr)

    log.logger.info("Using GPU device index:%s" % gpu_index)
    while True:
        try:
            message = socket.recv_json()
            inputs = message['inputs']
            module_name = message['module_name']
            inputs.update(modules_info[module_name]['predict_args'])
            inputs.update({'use_gpu': True})
            method = modules_info[module_name]['serving_method']
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
            output = method(**inputs)

        except Exception as err:
            log.logger.error(traceback.format_exc())
            output = package_result("101", str(err), "")
        socket.send_json(output)


if __name__ == '__main__':
    argv = sys.argv
    modules_info = json.loads(argv[1])
    gpu_index = argv[2]
    addr = argv[3]

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_index
    import paddlehub as hub
    from paddlehub.serving.http_server import package_result
    from paddlehub.utils import log

    filename = 'HubServing-%s.log' % time.strftime("%Y_%m_%d", time.localtime())
    logger = log.get_file_logger(filename)
    logger.logger.handlers = logger.logger.handlers[0:1]

    modules_pred_info = {}
    for module_name, module_info in modules_info.items():
        init_args = module_info.get('init_args', {})
        init_args.update({'name': module_name})
        module = hub.Module(**init_args)
        method_name = module.serving_func_name
        serving_method = getattr(module, method_name)
        predict_args = module_info.get('predict_args', {})
        modules_pred_info.update({module_name: {'predict_args': predict_args, 'serving_method': serving_method}})

    run_worker(modules_pred_info, gpu_index, addr)
