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

nlp_module_method = {
    "lac": "predict_lexical_analysis",
    "simnet_bow": "predict_sentiment_analysis",
    "lm_lstm": "predict_pretrained_model",
    "senta_lstm": "predict_pretrained_model",
    "senta_gru": "predict_pretrained_model",
    "senta_cnn": "predict_pretrained_model",
    "senta_bow": "predict_pretrained_model",
    "senta_bilstm": "predict_pretrained_model",
    "emotion_detection_textcnn": "predict_pretrained_model"
}
cv_module_method = {
    "vgg19_imagenet": "predict_classification",
    "vgg16_imagenet": "predict_classification",
    "vgg13_imagenet": "predict_classification",
    "vgg11_imagenet": "predict_classification",
    "shufflenet_v2_imagenet": "predict_classification",
    "se_resnext50_32x4d_imagenet": "predict_classification",
    "se_resnext101_32x4d_imagenet": "predict_classification",
    "resnet_v2_50_imagenet": "predict_classification",
    "resnet_v2_34_imagenet": "predict_classification",
    "resnet_v2_18_imagenet": "predict_classification",
    "resnet_v2_152_imagenet": "predict_classification",
    "resnet_v2_101_imagenet": "predict_classification",
    "pnasnet_imagenet": "predict_classification",
    "nasnet_imagenet": "predict_classification",
    "mobilenet_v2_imagenet": "predict_classification",
    "googlenet_imagenet": "predict_classification",
    "alexnet_imagenet": "predict_classification",
    "yolov3_coco2017": "predict_object_detection",
    "ultra_light_fast_generic_face_detector_1mb_640":
    "predict_object_detection",
    "ultra_light_fast_generic_face_detector_1mb_320":
    "predict_object_detection",
    "ssd_mobilenet_v1_pascal": "predict_object_detection",
    "pyramidbox_face_detection": "predict_object_detection",
    "faster_rcnn_coco2017": "predict_object_detection",
    "cyclegan_cityscapes": "predict_gan",
    "deeplabv3p_xception65_humanseg": "predict_semantic_segmentation",
    "ace2p": "predict_semantic_segmentation"
}


def predict_sentiment_analysis(module, input_text, batch_size, extra=None):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = input_text[0]
        data.update(input_text[1])
        results = predict_method(
            data=data, use_gpu=use_gpu, batch_size=batch_size)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_pretrained_model(module, input_text, batch_size, extra=None):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        data = {"text": input_text}
        results = predict_method(
            data=data, use_gpu=use_gpu, batch_size=batch_size)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_lexical_analysis(module, input_text, batch_size, extra=[]):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    data = {"text": input_text}
    try:
        if extra == []:
            results = predict_method(
                data=data, use_gpu=use_gpu, batch_size=batch_size)
        else:
            user_dict = extra[0]
            results = predict_method(
                data=data,
                user_dict=user_dict,
                use_gpu=use_gpu,
                batch_size=batch_size)
            for path in extra:
                os.remove(path)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_classification(module, input_img, batch_size):
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(
            data=input_img, use_gpu=use_gpu, batch_size=batch_size)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return {"result": "Please check data format!"}
    return results


def predict_gan(module, input_img, id, batch_size, extra={}):
    # special
    output_folder = module.name.split("_")[0] + "_" + "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(
            data=input_img, use_gpu=use_gpu, batch_size=batch_size)
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
            results[index] = results[index].replace(id + "_", "")
            results[index] = {"path": results[index]}
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        os.remove(os.path.join(output_folder, item))
    return results_pack


def predict_object_detection(module, input_img, id, batch_size):
    output_folder = "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(
            data=input_img, use_gpu=use_gpu, batch_size=batch_size)
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
            results[index]["path"] = results[index]["path"].replace(
                id + "_", "")
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        os.remove(os.path.join(output_folder, item))
    return results_pack


def predict_semantic_segmentation(module, input_img, id, batch_size):
    # special
    output_folder = module.name.split("_")[-1] + "_" + "output"
    global use_gpu
    method_name = module.desc.attr.map.data['default_signature'].s
    predict_method = getattr(module, method_name)
    try:
        input_img = {"image": input_img}
        results = predict_method(
            data=input_img, use_gpu=use_gpu, batch_size=batch_size)
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
        output_file_path = ""
        with open(results[index]["processed"], "rb") as fp:
            # special
            b_head = "data:image/png;base64"
            b_body = base64.b64encode(fp.read())
            b_body = str(b_body).replace("b'", "").replace("'", "")
            b_img = b_head + "," + b_body
            base64_list.append(b_img)
            output_file_path = results[index]["processed"]
            results[index]["origin"] = results[index]["origin"].replace(
                id + "_", "")
            results[index]["processed"] = results[index]["processed"].replace(
                id + "_", "")
            results[index].update({"base64": b_img})
            results_pack.append(results[index])
        os.remove(item)
        if output_file_path != "":
            os.remove(output_file_path)
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
        global use_gpu, batch_size_dict
        img_base64 = request.form.getlist("image")
        file_name_list = []
        if img_base64 != []:
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
        predict_func_name = cv_module_method.get(module_name, "")
        if predict_func_name != "":
            predict_func = eval(predict_func_name)
        else:
            module_type = module.type.split("/")[-1].replace("-", "_").lower()
            predict_func = eval("predict_" + module_type)
        batch_size = batch_size_dict.get(module_name, 1)
        results = predict_func(module, file_name_list, req_id, batch_size)
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
        predict_func_name = nlp_module_method.get(module_name, "")
        if predict_func_name != "":
            predict_func = eval(predict_func_name)
        else:
            module_type = module.type.split("/")[-1].replace("-", "_").lower()
            predict_func = eval("predict_" + module_type)
        file_list = []
        for item in file:
            file_path = req_id + "_" + item.filename
            file_list.append(file_path)
            item.save(file_path)
        batch_size = batch_size_dict.get(module_name, 1)
        results = predict_func(module, data, batch_size, file_list)
        return {"results": results}

    return app_instance


def config_with_file(configs):
    global nlp_module, cv_module, batch_size_dict
    nlp_module = []
    cv_module = []
    batch_size_dict = {}
    for item in configs:
        print(item)
        if item["category"] == "CV":
            cv_module.append(item["module"])
        elif item["category"] == "NLP":
            nlp_module.append(item["module"])
        batch_size_dict.update({item["module"]: item["batch_size"]})


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
