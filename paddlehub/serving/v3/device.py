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
import os
import traceback
import subprocess


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
            traceback.print_exc()
        finally:
            self.frontend.close()
            self.backend.close()
            context.term()


def start_workers(modules_name: list, gpus: list, backend_addr: str):
    '''
    InferenceWorker class provides workers for different GPU device.

    Args:
        modules_name(list): modules name
        gpus(list): GPU devices index
        backend_addr(str): the port of PaddleHub-Serving zmq backend address

    Examples:
    .. code-block:: python

        modules_name = ['lac', 'yolov3_darknet53_coco2017']
        start_workers(modules_name, ['0', '1', '2'], 'ipc://backend.ipc')

    '''
    modules_str = ''
    for module_name in modules_name:
        modules_str += ',%s' % module_name
    work_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'worker.py')
    for index in range(len(gpus)):
        subprocess.Popen(['python', work_file, modules_str[1:], gpus[index], backend_addr])


class InferenceServer(object):
    '''
    InferenceServer class starts zmq.rep as backend.

    Args:
        modules_name(list): modules name
        gpus(list): GPU devices index
    '''

    def __init__(self, modules_name: list, gpus: list):
        self.modules_name = modules_name
        self.gpus = gpus

    def listen(self, port: int):
        backend = "ipc://backend.ipc"
        start_workers(modules_name=self.modules_name, gpus=self.gpus, backend_addr=backend)
        d = InferenceDevice()
        d.listen('tcp://*:%s' % port, backend)
