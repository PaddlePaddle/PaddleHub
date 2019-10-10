# coding:utf8
from flask import Flask, request, render_template
from paddlehub.serving.model_service.text_model_service import TextModelService
from paddlehub.serving.model_service.image_model_service import ImageModelService
# from model_service.text_model_service import TextModelService
# from model_service.image_model_service import ImageModelService
import json
import hashlib
import time
import os
import base64
import logging
import sys

use_gpu = False


def create_app():
    # app_instance = Flask("paddlehub.serving.app")
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.route("/predict/image/<module_name>", methods=["POST"])
    def predict_iamge(module_name):
        global use_gpu
        img_base64 = request.form.get("input_img", "")
        received_file_name = request.form.get("input_file", "")
        md5_get = hashlib.md5()
        md5_src = str(time.time()) + str(img_base64)
        md5_get.update(md5_src.encode("utf-8"))

        ext = received_file_name.split(".")[-1]
        if ext == "":
            return {"result": "未识别的文件类型"}
        filename = md5_get.hexdigest() + "." + ext
        base64_head = img_base64.split(',')[0]
        img_data = base64.b64decode(img_base64.split(',')[-1])
        with open(filename, "wb") as fp:
            fp.write(img_data)
        input_data = {"image": [filename]}

        module = ImageModelService.get_module(module_name)
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            print(input_data)
            try:
                print("Use gpu is", use_gpu)
                results = predict_method(data=input_data, use_gpu=use_gpu)
            except Exception as err:
                print(err)
                return {"result": "请检查数据格式!"}
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)
        os.remove(filename)
        output_file = os.path.join("./output", filename)
        if output_file is not None and os.path.exists(output_file):
            with open(output_file, "rb") as fp:
                output_img_base64 = base64.b64encode(fp.read())
            os.remove(output_file)
        if module.type.startswith("CV"):
            results = {
                "border": str(results[0]["data"]),
                "output_img": base64_head + ","
                + str(output_img_base64).replace("b'", "").replace("'", "")
            }
        return {"result": results}

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        global use_gpu
        data = request.form.get("input_text", "")
        data = data.splitlines()
        data_temp = []
        for index in range(len(data)):
            data[index] = data[index].strip()
            if data[index] != "":
                print("gll")
                data_temp.append(data[index])
        data = data_temp
        if not isinstance(data, list):
            data = [data]
        input_data = {"text": data}

        module = TextModelService.get_module(module_name)
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            print(input_data)
            try:
                print("Use gpu is", use_gpu)
                results = predict_method(data=input_data, use_gpu=use_gpu)
            except Exception as err:
                print(err)
                return {"result": "请检查数据格式!"}
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)

        if results == []:
            results = "输入不能为空"
        return {"result": results}

    @app_instance.route("/predict/<module_name>", methods=["POST"])
    def predict(module_name):
        print(request.form)
        data = request.form.get("input_text", "")
        img_base64 = request.form.get("input_img", "")
        received_file_name = request.form.get("input_file", "")
        filename = None
        if module_name != "" and data != "":
            print("ininin")
            data = data.splitlines()
            print(data)
            data_temp = []
            for index in range(len(data)):
                data[index] = data[index].strip()
                if data[index] != "":
                    print("gll")
                    data_temp.append(data[index])
            data = data_temp
            if not isinstance(data, list):
                data = [data]
            input_data = {"text": data}
        # 如果是图片
        elif module_name != "" and img_base64 != "":
            md5_get = hashlib.md5()
            md5_src = str(time.time()) + str(img_base64)
            md5_get.update(md5_src.encode("utf-8"))
            ext = received_file_name.split(".")[-1]
            if ext == "":
                return {"result": "未识别的文件类型，请添加后缀"}
            filename = md5_get.hexdigest() + "." + ext
            base64_head = img_base64.split(',')[0]
            img_data = base64.b64decode(img_base64.split(',')[-1])
            print("123123123")
            print(img_data)
            with open(filename, "wb") as fp:
                fp.write(img_data)
            input_data = {"image": [filename]}
        else:
            return {"result": "不支持的文件类型！"}
        module = TextModelService.get_module(module_name)
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            print(input_data)
            try:
                results = predict_method(data=input_data)
            except Exception as err:
                print(err)
                return {"result": "请检查数据格式!"}
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)
        output_file = None
        if filename is not None:
            os.remove(filename)
            output_file = os.path.join("./output", filename)
        print("output_file=", output_file)

        if output_file is not None and os.path.exists(output_file):
            with open(output_file, "rb") as fp:
                output_img_base64 = base64.b64encode(fp.read())
            os.remove(output_file)
        if module.type.startswith("CV"):
            results = {
                "border": str(results[0]["data"]),
                "output_img": base64_head + ","
                + str(output_img_base64).replace("b'", "").replace("'", "")
            }
        print(results)
        if results == []:
            results = "输入不能为空."
        return {"result": results}

    return app_instance


def run(is_use_gpu=False):
    global use_gpu
    use_gpu = is_use_gpu
    my_app = create_app()
    my_app.run(host="0.0.0.0", port=8888, debug=True)


if __name__ == "__main__":
    run(is_use_gpu=False)
