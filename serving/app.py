# coding:utf8
from flask import Flask, request, jsonify
from serving.model_service.text_model_service import TextModelService
import json
import six
import hashlib
import time
import cv2
import paddlehub as hub
import os

if six.PY2:
    import sys
    reload(sys)  # noqa
    sys.setdefaultencoding("UTF-8")


def create_app():
    app_instance = Flask(__name__)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return {"status": 0}

    @app_instance.route("/inference/<module_name>", methods=["GET", "POST",
                                                             "PUT"])
    def inference(module_name):
        module = TextModelService.get_module(module_name)
        method = module.desc.attr.map.data['default_signature'].s
        if method == "":
            return {"Error": "The module can't be use for predict."}
        if request.method == "GET":
            data = request.args.get("input_text")
            if not isinstance(data, list):
                data = [data]
            input_data = {"text": data}
        elif request.method == "POST":
            data = json.loads(request.data)["input_text"]
            if not isinstance(data, list):
                data = [data]
            input_data = {"text": data}
        elif request.method == "PUT":
            file = request.files["input_file"]
            ext = file.filename.split(".")[-1]
            if ext == "":
                return {"result": "未识别的文件类型，请添加后缀"}
            md5_get = hashlib.md5()
            md5_src = str(time.time()) + file.filename
            md5_get.update(md5_src)
            filename = md5_get.hexdigest()
            if not file:
                return {"result": "上传失败"}
            filename = filename + "." + ext
            print("文件名是", filename)
            file.save(filename)
            filename = [filename]
            if file.mimetype.startswith("image"):
                print(file.mimetype)
                print("图片")
                input_data = {"image": filename}
            elif file.mimetype.startswith("text"):
                print("文本")

                with open(filename[0], "r") as fp:
                    contents = fp.read()
                    print(contents)
                    content_lines = contents.splitlines()
                data = []
                for sentence in content_lines:
                    if sentence != "":
                        data.append(sentence)
                print(data)
                input_data = {"text": data}
            else:
                return {"result": "不支持的文件类型！"}
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            print(input_data)
            print(type(input_data))
            # input_data = {'image': [u'1e647f8c4ef627b41941fd32676105dc.jpg']}
            # print(input_data)
            print(type(input_data))
            results = predict_method(data=input_data)
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)
        return {"result": results}

    return app_instance


def run():

    my_app = create_app()
    my_app.run(host="127.0.0.1", port=8888, debug=True)


def test_post(data):
    from serving.model_service.lac_model import lac_module_service
    inputs = {"text": data}
    result = lac_module_service.inference(inputs)
    return result


if __name__ == "__main__":
    mo = hub.Module("yolov3_coco2017")
    input_data = {'image': [u'1e647f8c4ef627b41941fd32676105dc.jpg']}
    print(os.getcwd())
    results = mo.object_detection(data=input_data)
    print(results)
    # run()
