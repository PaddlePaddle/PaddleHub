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

import time
import os
import multiprocessing
import platform

from flask import Flask, request

from paddlehub.serving.device import InferenceServer
from paddlehub.serving.client import InferenceClientProxy
from paddlehub.utils import utils, log

filename = 'HubServing-%s.log' % time.strftime("%Y_%m_%d", time.localtime())

if platform.system() == "Windows":

    class StandaloneApplication(object):
        def __init__(self):
            pass

        def load_config(self):
            pass

        def load(self):
            pass
else:
    import gunicorn.app.base

    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        '''
        StandaloneApplication class provides instance of StandaloneApplication
        as gunicorn backend.
        '''

        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super(StandaloneApplication, self).__init__()

        def load_config(self):
            config = {
                key: value
                for key, value in self.options.items() if key in self.cfg.settings and value is not None
            }
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application


def package_result(status: str, msg: str, data: dict):
    '''
    Package message of response.

    Args:
         status(str): Error code
            ========   ==============================================================================================
            Code       Meaning
            --------   ----------------------------------------------------------------------------------------------
            '000'      Return results normally
            '101'      An error occurred in the predicting method
            '111'      Module is not available
            '112'      Use outdated and abandoned HTTP protocol format
            ========   ===============================================================================================
         msg(str): Detailed info for error
         data(dict): Result of predict api.

    Returns:
        dict: Message of response

    Examples:
        .. code-block:: python

            data = {'result': 0.002}
            package_result(status='000', msg='', data=data)
    '''
    return {"status": status, "msg": msg, "results": data}


def create_app(client_port: int = 5559, modules_name: list = []):
    '''
    Start one flask instance and ready for HTTP requests.

    Args:
         client_port(str): port of zmq backend address
         modules_name(list): the name list of modules

    Returns:
        One flask instance.

    Examples:
        .. code-block:: python

            create_app(client_port='5559')
    '''
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    app_instance.logger = log.get_file_logger(filename)
    pid = os.getpid()

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        '''
        Provide index page.
        '''
        return '暂不提供可视化界面，请直接使用脚本进行请求。<br/>No visual ' \
               'interface is provided for the time being, please use the' \
               ' python script to make a request directly.'

    @app_instance.before_request
    def before_request():
        '''
        Add id info to `request.data` before request.
        '''
        request.data = {"id": utils.md5(request.remote_addr + str(time.time()))}

    @app_instance.route("/predict/<module_name>", methods=["POST"])
    def predict_serving_v3(module_name: str):
        '''
        Http api for predicting.

        Args:
            module_name(str): Module name for predicting.

        Returns:
            Result of predicting after packaging.
        '''

        if module_name not in modules_name:
            msg = "Module {} is not available.".format(module_name)
            return package_result("111", "", msg)
        inputs = request.json
        if inputs is None:
            results = "This usage is out of date, please use 'application/json' as content-type to post to /predict/%s. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details." % (
                module_name)
            return package_result("112", results, "")
        inputs = {'module_name': module_name, 'inputs': inputs}
        port_str = 'tcp://localhost:%s' % client_port

        client = InferenceClientProxy.get_client(pid, port_str)

        results = client.send_req(inputs)

        return package_result("000", results, "")

    return app_instance


def run(port: int = 8866, client_port: int = 5559, names: list = [], workers: int = 1):
    '''
    Run flask instance for PaddleHub-Serving

    Args:
         port(int): the port of the webserver
         client_port(int): the port of zmq backend address
         names(list): the name list of modules
         workers(int): workers for every client

    Examples:
        .. code-block:: python

            run(port=8866, client_port='5559')
    '''
    if platform.system() == "Windows":
        my_app = create_app(client_port, modules_name=names)
        my_app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
    else:
        options = {"bind": "0.0.0.0:%s" % port, "workers": workers, "worker_class": "sync"}
        StandaloneApplication(create_app(client_port, modules_name=names), options).run()

    log.logger.info("PaddleHub-Serving has been stopped.")


def run_http_server(port: int = 8866, client_port: int = 5559, names: list = [], workers: int = 1):
    '''
    Start subprocess to run function `run`

    Args:
        port(int): the port of the webserver
        client_port(int): the port of zmq backend address
        names(list): the name list of moduels
        workers(int): the workers for every client

    Returns:
        process id of subprocess

    Examples:
        .. code-block:: python

            run_http_server(port=8866, client_port='5559', names=['lac'])
    '''
    names = list(names)
    p = multiprocessing.Process(target=run, args=(port, client_port, names, workers))
    p.start()
    return p.pid


def run_all(modules_info: dict, gpus: list, frontend_port: int, backend_port: int):
    '''
    Run flask instance for frontend HTTP request and zmq device for backend zmq
    request.

    Args:
        modules_info(dict): modules info, include module name, version
        gpus(list): GPU devices index
        frontend_port(int): the port of PaddleHub-Serving frontend address
        backend_port(int): the port of PaddleHub-Serving zmq backend address

    Examples:
        .. code-block:: python

            modules_info = {'lac': {'init_args': {'version': '2.1.0'},
                                    'predict_args': {'batch_size': 1}}}
            run_all(modules_info, ['0', '1', '2'], 8866, 8867)
    '''
    run_http_server(frontend_port, backend_port, modules_info.keys(), len(gpus))
    MyIS = InferenceServer(modules_info, gpus)
    MyIS.listen(backend_port)
