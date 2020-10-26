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


def gen_result(status, msg, data):
    return {"status": status, "msg": msg, "results": data}


def run_worker(modules_info, gpu_index, addr):
    context = zmq.Context(1)
    socket = context.socket(zmq.REP)
    socket.connect(addr)

    print('Using GPU device index:', gpu_index)
    os.environ['DEVICE_INDEX'] = gpu_index
    while True:
        try:
            message = socket.recv_json()
            inputs = message['inputs']
            import paddlehub as hub
            module = hub.Module(name='lac')

            method_name = module.serving_func_name
            if method_name is None:
                raise RuntimeError("{} cannot be use for " "predicting".format(module))
                exit(1)
            serving_method = getattr(module, method_name)
            methood = serving_method
            output = methood(**inputs)

        except Exception as err:
            traceback.print_exc()
            output = gen_result("-1", "Please check data format!", "")
        socket.send_json(output)


class InferenceDevice(object):
    def __init__(self):
        self.frontend = None
        self.backend = None

    def listen(self, frontend_addr, backend_addr):
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


class InferenceWorker(object):
    def __init__(self, modules_name, gpus, backend_addr):
        self.modules_name = modules_name
        self.gpus = gpus
        self.backend_addr = backend_addr
        self.process = []
        self.modules_str = ''
        for module_name in self.modules_name:
            self.modules_str += ',%s' % module_name
        work_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'worker.py')
        for index in range(len(self.gpus)):
            subprocess.Popen(['python', work_file, self.modules_str[1:], self.gpus[index], self.backend_addr])

    def listen(self):
        for p in self.process:
            p.start()

    def term(self):
        for p in self.process:
            p.terminate()


class InferenceServer(object):
    def __init__(self, modules_name, gpus):
        self.modules_name = modules_name
        self.gpus = gpus

    def listen(self, port):
        backend = "ipc://backend.ipc"
        iw = InferenceWorker(modules_name=self.modules_name, gpus=self.gpus, backend_addr=backend)
        iw.listen()
        d = InferenceDevice()
        d.listen('tcp://*:%s' % port, backend)
