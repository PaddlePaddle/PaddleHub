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


class InferenceClient(object):
    def __init__(self, frontend_addr):
        self.frontend_addr = frontend_addr
        self.context = zmq.Context(1)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(frontend_addr)

    def send_req(self, message):
        self.socket.send_json(message)
        result = self.socket.recv_json()

        return result


class InferenceClientProxy(object):
    clients = {}

    @staticmethod
    def get_client(pid, frontend_addr):
        if pid not in InferenceClientProxy.clients.keys():
            client = InferenceClient(frontend_addr)
            InferenceClientProxy.clients.update({pid: client})
        return InferenceClientProxy.clients[pid]
