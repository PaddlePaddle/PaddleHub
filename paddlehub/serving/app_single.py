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
from paddlehub.serving.model_service.base_model_service import cv_module_info
from paddlehub.serving.model_service.base_model_service import nlp_module_info
from paddlehub.serving.model_service.base_model_service import v2_module_info
from paddlehub.common import utils
import functools
import time
import os
import base64
import logging
import glob


def gen_result(status, msg, data):
    return {"status": status, "msg": msg, "results": data}


def predict_v2_advanced(module_info, input):
    serving_method_name = module_info["method_name"]
    serving_method = getattr(module_info["module"], serving_method_name)
    predict_args = module_info["predict_args"].copy()
    predict_args.update(input)

    for item in serving_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    try:
        output = serving_method(**predict_args)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    return gen_result("0", "", output)


def predict_nlp(module_info, input_text, req_id, extra=None):
    method_name = module_info["method_name"]
    predict_method = getattr(module_info["module"], method_name)

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": input_text})
    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})

    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})

    if module_info["name"] == "lac" and extra.get("user_dict", []) != []:
        predict_args.update({"user_dict": extra.get("user_dict", [])[0]})

    try:
        res = predict_method(**predict_args)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        user_dict = extra.get("user_dict", [])
        for item in user_dict:
            if os.path.exists(item):
                os.remove(item)
    return gen_result("0", "", res)


def predict_classification(module_info, input_img, id, extra={}):
    method_name = module_info["method_name"]
    module = module_info["module"]
    predict_method = getattr(module, method_name)

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": {"image": input_img}})
    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})
    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    try:
        results = predict_method(**predict_args)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        for item in input_img:
            if os.path.exists(item):
                os.remove(item)
    return gen_result("0", "", str(results))


def predict_gan(module_info, input_img, id, extra={}):
    method_name = module_info["method_name"]
    module = module_info["module"]
    predict_method = getattr(module, method_name)

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": {"image": input_img}})
    predict_args["data"].update(extra)
    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})
    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    results = predict_method(**predict_args)
    try:
        pass
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        base64_list = []
        results_pack = []
        for index in range(len(input_img)):
            item = input_img[index]
            output_file = results[index].split(" ")[-1]
            with open(output_file, "rb") as fp:
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
            os.remove(output_file)
    return gen_result("0", "", str(results_pack))


def predict_mask(module_info, input_img, id, extra=None, r_img=False):
    output_folder = "detection_result"
    method_name = module_info["method_name"]
    module = module_info["module"]
    predict_method = getattr(module, method_name)
    data_len = len(input_img) if input_img is not None else 0
    data = {}
    if input_img is not None:
        input_img = {"image": input_img}
        data.update(input_img)
    if extra is not None:
        data.update(extra)
        r_img = True if "visual_result" in extra.keys() else False

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": data})
    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})
    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    try:
        results = predict_method(**predict_args)
        results = utils.handle_mask_results(results, data_len)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        base64_list = []
        results_pack = []
        if input_img is not None:
            if r_img is False:
                for index in range(len(results)):
                    results[index]["path"] = ""
                results_pack = results
                str_id = id + "*"
                files_deleted = glob.glob(str_id)
                for path in files_deleted:
                    if os.path.exists(path):
                        os.remove(path)
            else:
                input_img = input_img.get("image", [])
                for index in range(len(input_img)):
                    item = input_img[index]
                    file_path = os.path.join(output_folder, item)
                    if not os.path.exists(file_path):
                        results_pack.append(results[index])
                        os.remove(item)
                    else:
                        with open(file_path, "rb") as fp:
                            b_head = "data:image/" + item.split(
                                ".")[-1] + ";base64"
                            b_body = base64.b64encode(fp.read())
                            b_body = str(b_body).replace("b'", "").replace(
                                "'", "")
                            b_img = b_head + "," + b_body
                            base64_list.append(b_img)
                            results[index]["path"] = results[index]["path"].replace(
                                id + "_", "") if results[index]["path"] != "" \
                                else ""

                            results[index].update({"base64": b_img})
                            results_pack.append(results[index])
                        os.remove(item)
                        os.remove(os.path.join(output_folder, item))
        else:
            results_pack = results

    return gen_result("0", "", str(results_pack))


def predict_object_detection(module_info, input_img, id, extra={}):
    output_folder = "detection_result"
    method_name = module_info["method_name"]
    module = module_info["module"]
    predict_method = getattr(module, method_name)

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": {"image": input_img}})

    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})
    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    try:
        results = predict_method(**predict_args)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        base64_list = []
        results_pack = []
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
    return gen_result("0", "", str(results_pack))


def predict_semantic_segmentation(module_info, input_img, id, extra={}):
    method_name = module_info["method_name"]
    module = module_info["module"]
    predict_method = getattr(module, method_name)

    predict_args = module_info["predict_args"].copy()
    predict_args.update({"data": {"image": input_img}})

    if isinstance(predict_method, functools.partial):
        predict_method = predict_method.func
        predict_args.update({"sign_name": method_name})
    for item in predict_method.__code__.co_varnames:
        if item in module_info.keys():
            predict_args.update({item: module_info[item]})
    try:
        results = predict_method(**predict_args)
    except Exception as err:
        curr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(curr, " - ", err)
        return gen_result("-1", "Please check data format!", "")
    finally:
        base64_list = []
        results_pack = []
        for index in range(len(input_img)):
            item = input_img[index]
            with open(results[index]["processed"], "rb") as fp:
                b_head = "data:image/png;base64"
                b_body = base64.b64encode(fp.read())
                b_body = str(b_body).replace("b'", "").replace("'", "")
                b_img = b_head + "," + b_body
                base64_list.append(b_img)
                output_file_path = results[index]["processed"]
                results[index]["origin"] = results[index]["origin"].replace(
                    id + "_", "")
                results[index]["processed"] = results[index][
                    "processed"].replace(id + "_", "")
                results[index].update({"base64": b_img})
                results_pack.append(results[index])
            os.remove(item)
            if output_file_path != "":
                os.remove(output_file_path)
    return gen_result("0", "", str(results_pack))


def create_app(init_flag=False, configs=None):
    if init_flag is False:
        if configs is None:
            raise RuntimeError("Lack of necessary configs.")
        config_with_file(configs)

    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    logging.basicConfig()
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.before_request
    def before_request():
        request.data = {"id": utils.md5(request.remote_addr + str(time.time()))}

    @app_instance.route("/get/modules", methods=["GET", "POST"])
    def get_modules_info():
        module_info = {}
        if len(nlp_module_info.nlp_modules) > 0:
            module_info.update({"nlp_module": [{"Choose...": "Choose..."}]})
            for item in nlp_module_info.nlp_modules:
                module_info["nlp_module"].append({item: item})
        if len(cv_module_info.cv_modules) > 0:
            module_info.update({"cv_module": [{"Choose...": "Choose..."}]})
            for item in cv_module_info.cv_modules:
                module_info["cv_module"].append({item: item})
        return {"module_info": module_info}

    @app_instance.route("/predict/image/<module_name>", methods=["POST"])
    def predict_image(module_name):
        if request.path.split("/")[-1] not in cv_module_info.modules_info:
            return {"error": "Module {} is not available.".format(module_name)}
        module_info = cv_module_info.get_module_info(module_name)
        if module_info["code_version"] == "v2":
            results = {}
            # results = predict_v2(module_info, inputs)
            results.update({
                "Warnning":
                "This usage is out of date, please "
                "use 'application/json' as "
                "content-type to post to "
                "/predict/%s. See "
                "'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details."
                % (module_name)
            })
            return gen_result("-1", results, "")
        req_id = request.data.get("id")
        img_base64 = request.form.getlist("image")
        extra_info = {}
        for item in list(request.form.keys()):
            extra_info.update({item: request.form.getlist(item)})

        for key in extra_info.keys():
            if isinstance(extra_info[key], list):
                extra_info[key] = utils.base64s_to_cvmats(
                    eval(extra_info[key][0])["b64s"]) if isinstance(
                        extra_info[key][0], str
                    ) and "b64s" in extra_info[key][0] else extra_info[key]

        file_name_list = []
        if img_base64 != []:
            for item in img_base64:
                ext = item.split(";")[0].split("/")[-1]
                if ext not in ["jpeg", "jpg", "png"]:
                    return gen_result("-1", "Unrecognized file type", "")
                filename = req_id + "_" \
                           + utils.md5(str(time.time()) + item[0:20]) \
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

        module = module_info["module"]
        predict_func_name = cv_module_info.cv_module_method.get(module_name, "")
        if predict_func_name != "":
            predict_func = eval(predict_func_name)
        else:
            module_type = module.type.split("/")[-1].replace("-", "_").lower()
            predict_func = eval("predict_" + module_type)
        if file_name_list == []:
            file_name_list = None
        if extra_info == {}:
            extra_info = None
        results = predict_func(module_info, file_name_list, req_id, extra_info)

        return results

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        if request.path.split("/")[-1] not in nlp_module_info.nlp_modules:
            return {"error": "Module {} is not available.".format(module_name)}
        module_info = nlp_module_info.get_module_info(module_name)
        if module_info["code_version"] == "v2":
            results = "This usage is out of date, please use 'application/json' as content-type to post to /predict/%s. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details." % (
                module_name)
            return gen_result("-1", results, "")
        req_id = request.data.get("id")
        inputs = {}
        for item in list(request.form.keys()):
            inputs.update({item: request.form.getlist(item)})
        files = {}
        for file_key in list(request.files.keys()):
            files[file_key] = []
            for file in request.files.getlist(file_key):
                file_name = req_id + "_" + file.filename
                files[file_key].append(file_name)
                file.save(file_name)

        results = predict_nlp(
            module_info=module_info,
            input_text=inputs,
            req_id=req_id,
            extra=files)
        return results

    @app_instance.route("/predict/<module_name>", methods=["POST"])
    def predict_modulev2(module_name):
        if module_name in v2_module_info.modules:
            module_info = v2_module_info.get_module_info(module_name)
        else:
            msg = "Module {} is not available.".format(module_name)
            return gen_result("-1", msg, "")
        inputs = request.json
        if inputs is None:
            results = "This usage is out of date, please use 'application/json' as content-type to post to /predict/%s. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details." % (
                module_name)
            return gen_result("-1", results, "")

        results = predict_v2_advanced(module_info, inputs)
        return results

    return app_instance


def config_with_file(configs):
    for key, value in configs.items():
        if "CV" == value["category"]:
            cv_module_info.add_module(key, {key: value})
        elif "NLP" == value["category"]:
            nlp_module_info.add_module(key, {key: value})
        v2_module_info.add_module(key, {key: value})
        print(key, "==", value["version"])


def run(configs=None, port=8866):
    if configs is not None:
        config_with_file(configs)
    else:
        print("Start failed cause of missing configuration.")
        return
    my_app = create_app(init_flag=True)
    my_app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
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
