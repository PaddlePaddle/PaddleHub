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
from flask import Flask, request, render_template
from paddlehub.serving.model_service.text_model_service import TextModelService
from paddlehub.serving.model_service.image_model_service import ImageModelService
from paddlehub.common import utils
# from model_service.text_model_service import TextModelService
# from model_service.image_model_service import ImageModelService
import time
import os
import base64
import logging


def predict_sentiment_analysis(module, input_text, extra=None):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = input_text[0]
        data.update(input_text[1])
        results = predict_method(data=data, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_pretrained_model(module, input_text, extra=None):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = {"text": input_text}
        results = predict_method(data=data, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_lexical_analysis(module, input_text, extra=[]):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    data = {"text": input_text}
    try:
        if extra == []:
            results = predict_method(data=data, use_gpu=use_gpu)
        else:
            user_dict = extra[0]
            results = predict_method(
                data=data, user_dict=user_dict, use_gpu=use_gpu)
            for path in extra:
                os.remove(path)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_classification(module, input_img):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(data=input_img, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_gan(module, input_img, extra={}):
    # special
    output_folder = module.name.split("_")[0] + "_" + "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(data=input_img, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    base64_list = []
    results_pack = []
    input_img = input_img.get("image", [])
    for index in range(len(input_img)):
        # special
        item = input_img[index]
        with open(os.path.join(output_folder, item), "rb") as fp:
            # special
            b_head = "data:image/" + item.split(".")[-1] + ";base64"
            b_body = base64.b64encode(fp.read())
            b_body = str(b_body).replace("b'", "").replace("'", "")
            b_img = b_head + "," + b_body
            base64_list.append(b_img)
            results[index] = {"path": results[index]}
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        os.remove(os.path.join(output_folder, item))
    return results_pack


def predict_object_detection(module, input_img):
    output_folder = "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(data=input_img, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    base64_list = []
    results_pack = []
    input_img = input_img.get("image", [])
    for index in range(len(input_img)):
        item = input_img[index]
        with open(os.path.join(output_folder, item), "rb") as fp:
            b_head = "data:image/" + item.split(".")[-1] + ";base64"
            b_body = base64.b64encode(fp.read())
            b_body = str(b_body).replace("b'", "").replace("'", "")
            b_img = b_head + "," + b_body
            base64_list.append(b_img)
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        os.remove(os.path.join(output_folder, item))
    return results_pack


def predict_semantic_segmentation(module, input_img):
    # special
    output_folder = module.name.split("_")[-1] + "_" + "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(data=input_img, use_gpu=use_gpu)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    base64_list = []
    results_pack = []
    input_img = input_img.get("image", [])
    for index in range(len(input_img)):
        # special
        item = input_img[index]
        with open(results[index]["processed"], "rb") as fp:
            # special
            b_head = "data:image/png;base64"
            b_body = base64.b64encode(fp.read())
            b_body = str(b_body).replace("b'", "").replace("'", "")
            b_img = b_head + "," + b_body
            base64_list.append(b_img)
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        os.remove(results[index]["processed"])
    return results_pack


def create_app():
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.before_request
    def before_request():
        request.data = {"id": utils.md5(request.remote_addr + str(time.time()))}
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
    def predict_image(module_name):
        req_id = request.data.get("id")
        global use_gpu
        img_base64 = request.form.getlist("image")
        file_name_list = []
        if img_base64 != "":
            for item in img_base64:
                ext = item.split(";")[0].split("/")[-1]
                if ext not in ["jpeg", "jpg", "png"]:
                    return {"result": "Unrecognized file type"}
                filename = req_id + "_" \
                           + utils.md5(str(time.time())+item[0:20]) \
                           + "." \
                           + ext
                img_data = base64.b64decode(item.split(',')[-1])
                file_name_list.append(filename)
                with open(filename, "wb") as fp:
                    fp.write(img_data)
        else:
            file = request.files.getlist("image")
            for item in file:
                file_name = req_id + "_" + item.filename
                item.save(file_name)
                file_name_list.append(file_name)
        module = ImageModelService.get_module(module_name)
        module_type = module.type.split("/")[-1].replace("-", "_").lower()
        predict_func = eval("predict_" + module_type)
        results = predict_func(module, file_name_list)
        r = {"results": str(results)}
        return r

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        req_id = request.data.get("id")
        global use_gpu
        if module_name == "simnet_bow":
            text_1 = request.form.getlist("text_1")
            text_2 = request.form.getlist("text_2")
            data = [{"text_1": text_1}, {"text_2": text_2}]
        else:
            data = request.form.getlist("text")
        file = request.files.getlist("user_dict")
        module = TextModelService.get_module(module_name)
        module_type = module.type.split("/")[-1].replace("-", "_").lower()
        predict_func = eval("predict_" + module_type)
        file_list = []
        for item in file:
            file_path = req_id + "_" + item.filename
            file_list.append(file_path)
            item.save(file_path)
        results = predict_func(module, data, file_list)
        return {"results": results}

    return app_instance


def config_with_file(configs):
    global nlp_module, cv_module
    nlp_module = []
    cv_module = []
    for item in configs:
        print(item)
        if item["category"] == "CV":
            cv_module.append(item["module"])
        elif item["category"] == "NLP":
            nlp_module.append(item["module"])


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
