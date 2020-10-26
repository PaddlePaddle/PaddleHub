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

from flask import Flask, request
import multiprocessing
import paddlehub as hub
from paddlehub.serving.v3.device import InferenceServer, gen_result
from paddlehub.serving.v3.client import InferenceClient
from paddlehub.common import utils
import time
import logging
import subprocess


def create_app(client_port='5559'):
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    logging.basicConfig()
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)

    port_str = 'tcp://localhost:%s' % (client_port)
    client = InferenceClient(port_str)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return '暂不提供可视化界面，请直接使用脚本进行请求。<br/>No visual ' \
               'interface is provided for the time being, please use the' \
               ' python script to make a request directly.'

    @app_instance.before_request
    def before_request():
        request.data = {"id": utils.md5(request.remote_addr + str(time.time()))}

    @app_instance.route("/predict/<module_name>", methods=["POST"])
    def predict_modulev2(module_name):
        inputs = request.json
        if inputs is None:
            results = "This usage is out of date, please use 'application/json' as content-type to post to /predict/%s. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details." % (
                module_name)
            return gen_result("-1", results, "")
        inputs = {'module_name': module_name, 'inputs': inputs}

        results = client.send_req(inputs)

        return gen_result("-1", results, "")

    return app_instance


def run(port=8866, client_port='5559'):
    my_app = create_app(client_port)
    my_app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
    print("PaddleHub-Serving has been stopped.")


def run_http_server(port=8866, client_port='5559'):
    p = multiprocessing.Process(target=run, args=(port, client_port))
    p.start()
    return p.pid


def run_all(modules_name, gpus, frontend_port, backend_port):
    run_http_server(frontend_port, backend_port)
    MyIS = InferenceServer(modules_name, gpus)
    MyIS.listen(backend_port)


if __name__ == "__main__":
    modules_name = ['lac', 'yolov3_darknet53_coco2017']
    run_all(modules_name, ['0', '1', '2'], 8866, 8867)
    exit()
