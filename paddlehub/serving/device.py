# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import subprocess

from paddlehub.utils import log

filename = 'HubServing-%s.log' % time.strftime("%Y_%m_%d_%H%M%S", time.localtime())
logger = log.get_file_logger(filename)


class InferenceDevice(object):
    '''
    The InferenceDevice class provides zmq.device to connect with frontend and
    backend.
    '''

    def __init__(self):
        self.frontend = None
        self.backend = None

    def listen(self, frontend_addr: str, backend_addr: str):
        '''
        Start zmq.device to listen from frontend address to backend address.
        '''
        try:
            context = zmq.Context(1)

            self.frontend = context.socket(zmq.ROUTER)
            self.frontend.bind(frontend_addr)

            self.backend = context.socket(zmq.DEALER)
            self.backend.bind(backend_addr)

            zmq.device(zmq.QUEUE, self.frontend, self.backend)
        except Exception as e:
            logger.error(traceback.format_exc())
        finally:
            self.frontend.close()
            self.backend.close()
            context.term()


def start_workers(modules_info: dict, gpus: list, backend_addr: str):
    '''
    InferenceWorker class provides workers for different GPU device.

    Args:
        modules_info(dict): modules info, include module name, version
        gpus(list): GPU devices index
        backend_addr(str): the port of PaddleHub-Serving zmq backend address

    Examples:
    .. code-block:: python

        modules_info = {'lac': {'init_args': {'version': '2.1.0'},
                                'predict_args': {'batch_size': 1}}}
        start_workers(modules_name, ['0', '1', '2'], 'ipc://backend.ipc')

    '''
    work_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'worker.py')
    modules_info = json.dumps(modules_info)
    for index in range(len(gpus)):
        subprocess.Popen(['python', work_file, modules_info, gpus[index], backend_addr])


class InferenceServer(object):
    '''
    InferenceServer class starts zmq.rep as backend.

    Args:
        modules_name(list): modules name
        gpus(list): GPU devices index
    '''

    def __init__(self, modules_info: dict, gpus: list):
        self.modules_info = modules_info
        self.gpus = gpus

    def listen(self, port: int):
        backend = "ipc://backend.ipc"
        start_workers(modules_info=self.modules_info, gpus=self.gpus, backend_addr=backend)
        d = InferenceDevice()
        d.listen('tcp://*:%s' % port, backend)
