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


def get_img_output(module, base64_head, results):
    if module.type.startswith("CV"):
        if "semantic-segmentation" in module.type:
            output_file = results[0].get("processed", None)
            if output_file is not None and os.path.exists(output_file):
                with open(output_file, "rb") as fp:
                    output_img_base64 = base64.b64encode(fp.read())
                os.remove(output_file)
                results = {
                    "desc":
                    "Here is result.",
                    "output_img":
                    base64_head + "," + str(output_img_base64).replace(
                        "b'", "").replace("'", "")
                }
            return {"result": results}
        elif "object-detection" in module.type:
            output_file = os.path.join("./output", results[0]["path"])
            if output_file is not None and os.path.exists(output_file):
                with open(output_file, "rb") as fp:
                    output_img_base64 = base64.b64encode(fp.read())
                os.remove(output_file)
                results = {
                    "desc":
                    str(results[0]["data"]),
                    "output_img":
                    base64_head + "," + str(output_img_base64).replace(
                        "b'", "").replace("'", "")
                }
            return {"result": results}


def predict_sentiment_analysis(module, input_text):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = eval(input_text[0])
        data.update(eval(input_text[1]))
        results = predict_method(data=data, use_gpu=use_gpu)
    except Exception as err:
        return {"result": "Please check data format!"}
    return results


def predict_pretrained_model(module, input_text):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = {"text": input_text}
        results = predict_method(data=data, use_gpu=use_gpu)
    except Exception as err:
        return {"result": "Please check data format!"}
    return results


def predict_lexical_analysis(module, input_text, extra=None):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        if extra is None:
            data = {"text": input_text}
        else:
            data = {"text": input_text}
        results = predict_method(data=data, use_gpu=use_gpu)
    except Exception as err:
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
        # 稍后保存的文件名用id+源文件名的形式以避免冲突
        req_id = request.data.get("id")
        global use_gpu
        # 这里是一个base64的列表
        img_base64 = request.form.getlist("input_img")
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
            file = request.files.getlist("input_img")
            for item in file:
                file_name = req_id + "_" + item.filename
                item.save(file_name)
                file_name_list.append(file_name)
            # 到这里就把所有原始文件和文件名列表都保存了
            # 文件名列表可用于预测
            # 获取模型
        module = ImageModelService.get_module(module_name)
        # 根据模型种类寻找具体预测方法，即根据名字定函数
        module_type = module.type.split("/")[-1].replace("-", "_").lower()
        predict_func = eval("predict_" + module_type)
        results = predict_func(module, file_name_list)
        r = {"results": str(results)}
        return r

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        global use_gpu
        # 应该是一个列表
        data = request.form.getlist("input_text")
        module = TextModelService.get_module(module_name)
        module_type = module.type.split("/")[-1].replace("-", "_").lower()
        predict_func = eval("predict_" + module_type)
        results = predict_func(module, data)
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


def run(is_use_gpu=False, configs=None, port=8888, timeout=60):
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
