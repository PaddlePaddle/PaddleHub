# coding: utf-8
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
from flask import Flask, request, render_template
from paddlehub.serving.model_service.text_model_service import TextModelService
from paddlehub.serving.model_service.image_model_service import ImageModelService
from paddlehub.common import utils
import time
import os
import base64
import logging
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import random
import six
if six.PY2:
    from Queue import PriorityQueue
if six.PY3:
    from queue import PriorityQueue


class MyPriorityQueue(PriorityQueue):
    def get_attribute(self, name):
        return getattr(self, name)


class Manager(BaseManager):
    pass


Manager.register("get_priorityQueue", MyPriorityQueue)


def choose_module_category(input_data, module_name, batch_size=1):
    global nlp_module, cv_module
    if module_name in nlp_module:
        predict_nlp(input_data, module_name, batch_size)
    elif module_name in cv_module:
        predict_cv(input_data, module_name, batch_size)


def predict_nlp(input_data, module_name, batch_size=1):
    global use_gpu
    real_input_data = []
    for index in range(len(input_data)):
        real_input_data.append(input_data[index][3])
    module = TextModelService.get_module(module_name)
    method_name = module.desc.attr.map.data['default_signature'].s
    if method_name != "":
        predict_method = getattr(module, method_name)
        try:
            real_input_data = {"text": real_input_data}
            results = predict_method(
                data=real_input_data, use_gpu=use_gpu, batch_size=batch_size)

        except Exception as err:
            return {"result": "Please check data format!"}
    else:
        results = "Module {} can't be use for predicting.".format(module_name)
    try:
        result_data = []
        for index in range(len(input_data)):
            result_data.append(list(input_data[index]))
            result_data[-1][3] = results[index]
    except Exception as err:
        print("Transform error!")
    for index in range(len(result_data)):
        if results_dict.get(result_data[index][2]) is None:
            results_dict[result_data[index][2]] = [[
                result_data[index][1], result_data[index][3]
            ]]
        else:
            temp_list = results_dict[result_data[index][2]]
            temp_list.append([result_data[index][1], result_data[index][3]])
            results_dict[result_data[index][2]] = temp_list
    return {"result": results_dict}


def predict_cv(input_data, module_name, batch_size=1):
    global use_gpu
    filename_list = []
    for index in range(len(input_data)):
        filename_list.append(input_data[index][3])
    input_images = {"image": filename_list}
    module = ImageModelService.get_module(module_name)
    method_name = module.desc.attr.map.data['default_signature'].s
    if method_name != "":
        predict_method = getattr(module, method_name)
        try:
            results = predict_method(
                data={"image": filename_list},
                use_gpu=use_gpu,
                batch_size=batch_size)
        except Exception as err:
            return {"result": "Please check data format!"}
    else:
        results = "Module {} can't be use for predicting.".format(module_name)
    try:
        result_data = []
        for index in range(len(input_data)):
            result_data.append(list(input_data[index]))
            result_data[-1][3] = results[index]
    except Exception as err:
        print("Transform error!")
    for index in range(len(result_data)):
        if results_dict.get(result_data[index][2]) is None:
            results_dict[result_data[index][2]] = [[
                result_data[index][1], result_data[index][3]
            ]]
        else:
            temp_list = results_dict[result_data[index][2]]
            temp_list.append([result_data[index][1], result_data[index][3]])
            results_dict[result_data[index][2]] = temp_list
    return {"result": results}


def worker():
    global batch_size_list, name_list, queue_name_list, cv_module
    latest_num = random.randrange(0, len(queue_name_list))
    try:
        while True:
            time.sleep(0.01)
            for index in range(len(queue_name_list)):
                while queues_dict[
                        queue_name_list[latest_num]].empty() is not True:
                    input_data = []
                    lock.acquire()
                    try:
                        batch = queues_dict[
                            queue_name_list[latest_num]].get_attribute(
                                "maxsize")
                        for index2 in range(batch):
                            if queues_dict[queue_name_list[latest_num]].empty(
                            ) is True:
                                break
                            input_data.append(
                                queues_dict[queue_name_list[latest_num]].get())
                    finally:
                        lock.release()
                    if len(input_data) != 0:
                        choose_module_category(input_data,
                                               queue_name_list[latest_num],
                                               batch_size_list[latest_num])
                    else:
                        pass
                latest_num = (latest_num + 1) % len(queue_name_list)
    except KeyboardInterrupt:
        print("Process %s is end." % (os.getpid()))


def init_pool(l):
    global lock
    lock = l


def create_app():
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)
    global queues_dict, pool
    lock = mp.Lock()
    pool = mp.Pool(
        processes=(mp.cpu_count() - 1),
        initializer=init_pool,
        initargs=(lock, ))
    for i in range(mp.cpu_count() - 1):
        pool.apply_async(worker)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.before_request
    def before_request():
        request.data = {"id": utils.md5(request.remote_addr + str(time.time()))}
        print(request.remote_addr)
        pass

    @app_instance.route("/get/modules", methods=["GET", "POST"])
    def get_modules_info():
        global nlp_module, cv_module
        module_info = {}
        if len(nlp_module) > 0:
            module_info.update({"nlp_module": [{"Choose...": "Choose..."}]})
            for item in nlp_module:
                module_info["nlp_module"].append({item: item})
        if len(cv_module) > 0:
            module_info.update({"cv_module": [{"Choose...": "Choose..."}]})
            for item in cv_module:
                module_info["cv_module"].append({item: item})
        module_info.update({"Choose...": [{"请先选择分类": "Choose..."}]})
        return {"module_info": module_info}

    @app_instance.route("/predict/image/<module_name>", methods=["POST"])
    def predict_iamge(module_name):
        global results_dict
        req_id = request.data.get("id")

        img_base64 = request.form.get("image", "")
        if img_base64 != "":
            img_base64 = request.form.get("image", "")
            ext = img_base64.split(";")[0].split("/")[-1]
            if ext not in ["jpeg", "jpg", "png"]:
                return {"result": "Unrecognized file type"}
            filename = utils.md5(str(time.time()) + str(img_base64)) + "." + ext
            base64_head = img_base64.split(',')[0]
            img_data = base64.b64decode(img_base64.split(',')[-1])
            with open(filename, "wb") as fp:
                fp.write(img_data)
        else:
            file = request.files["image"]
            filename = file.filename
            ext = file.filename.split(".")[-1]
            if ext not in ["jpeg", "jpg", "png"]:
                return {"result": "Unrecognized file type"}
            base64_head = "data:image/" + ext + ";base64"
            filename = utils.md5(filename) + '.' + ext
            file.save(filename)
        score = time.time()
        file_list = [filename]
        if queues_dict[module_name].qsize(
        ) + 1 > queues_dict[module_name].get_attribute("maxsize"):
            return {"result": "Too many visitors now, please come back later."}
        data_2_item(file_list, req_id, score, module_name)
        data_num = len(file_list)
        results = []
        result_len = 0
        start_time = time.time()
        while result_len != data_num:
            result_len = len(results_dict.get(req_id, []))
            if time.time() - start_time > time_out:
                results_dict.pop(req_id, None)
                return {"result": "Request time out."}
        results = results_dict.get(req_id)
        results_dict.pop(req_id, None)
        results = [i[1] for i in sorted(results, key=lambda k: k[0])]
        filename = results[0].get("path")
        ext = filename.split(".")[-1]
        if filename is not None:
            output_file = os.path.join("./output", filename)
            if output_file is not None and os.path.exists(output_file):
                with open(output_file, "rb") as fp:
                    output_img_base64 = base64.b64encode(fp.read())
                os.remove(filename)
                os.remove(output_file)
                results = {
                    "desc":
                    str(results[0]["data"]),
                    "output_img":
                    base64_head + "," + str(output_img_base64).replace(
                        "b'", "").replace("'", "")
                }
                return {"result": results}
        return {"result": str(results)}

    def data_2_item(data_list, req_id, score, module_name):
        global queues_dict
        for index in range(len(data_list)):
            queues_dict[module_name].put((score, index, req_id,
                                          data_list[index]))

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        global results_dict, queues_dict
        req_id = request.data.get("id")
        data_list = request.form.get("text")
        score = time.time()
        data_list = data_list.splitlines()
        data_temp = []
        for index in range(len(data_list)):
            data_list[index] = data_list[index].strip()
            if data_list[index] != "":
                data_temp.append(data_list[index])
        data_list = data_temp
        if not isinstance(data_list, list):
            data_list = [data_list]
        data_num = len(data_list)
        if data_num > queues_dict[module_name].get_attribute("maxsize"):
            return {"result": ["Too much data, please reduce the data."]}
        if data_num + queues_dict[module_name].qsize(
        ) > queues_dict[module_name].get_attribute("maxsize"):
            return {"result": "Too many visitors now, please come back later."}
        data_2_item(data_list, req_id, score, module_name)
        results = []
        result_len = 0
        start_time = time.time()
        while result_len != data_num:
            result_len = len(results_dict.get(req_id, []))
            if time.time() - start_time > time_out:
                results_dict.pop(req_id, None)
                return {"result": "Request time out."}
        results = results_dict.get(req_id)
        results_dict.pop(req_id, None)
        results = [i[1] for i in sorted(results, key=lambda k: k[0])]
        return {"result": results}

    return app_instance


def config_with_file(configs):
    global m
    global nlp_module, cv_module, queues_list, batch_size_list, name_list, \
        queues_dict, queue_name_list, results_dict
    m = Manager()
    m.start()
    nlp_module = []
    cv_module = []
    queues_list = []
    batch_size_list = []
    name_list = []
    queues_dict = {}
    queue_name_list = []
    results_dict = mp.Manager().dict()
    for item in configs:
        print(item)
        if item["category"] == "CV":
            cv_module.append(item["module"])
        elif item["category"] == "NLP":
            nlp_module.append(item["module"])
        queues_list.append(m.get_priorityQueue(maxsize=item["queue_size"]))
        batch_size_list.append(item["batch_size"])
        name_list.append(item["module"])
        queues_dict.update({item["module"]: queues_list[-1]})
        queue_name_list.append(item["module"])


def run(is_use_gpu=False, configs=None, port=8866, timeout=60):
    global use_gpu, time_out
    time_out = timeout
    use_gpu = is_use_gpu
    if configs is not None:
        config_with_file(configs)
    else:
        print("Start failed cause of missing configuration.")
        return
    my_app = create_app()
    my_app.run(host="0.0.0.0", port=port, debug=False)
    pool.close()
    pool.join()
    print("PaddleHub-Serving has been stopped.")


if __name__ == "__main__":
    configs = [{
        'category': 'NLP',
        u'queue_size': 20,
        u'version': u'1.0.0',
        u'module': 'lac',
        u'batch_size': 20
    },
               {
                   'category': 'NLP',
                   u'queue_size': 20,
                   u'version': u'1.0.0',
                   u'module': 'senta_lstm',
                   u'batch_size': 20
               },
               {
                   'category': 'CV',
                   u'queue_size': 20,
                   u'version': u'1.0.0',
                   u'module': 'yolov3_coco2017',
                   u'batch_size': 20
               },
               {
                   'category': 'CV',
                   u'queue_size': 20,
                   u'version': u'1.0.0',
                   u'module': 'faster_rcnn_coco2017',
                   u'batch_size': 20
               }]
    run(is_use_gpu=False, configs=configs)
