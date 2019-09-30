# coding:utf8
from flask import Flask, request, jsonify, render_template
from serving.model_service.text_model_service import TextModelService
import json
import hashlib
import time
import os
import base64


def create_app():
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False

    @app_instance.route("/t", methods=["GET", "POST"])
    def test():
        return render_template("main.html")

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.route("/predict/<module_name>", methods=["POST"])
    def predict(module_name):
        print(request.form)
        data = request.form.get("input_text", "")
        img_base64 = request.form.get("input_img", "")
        received_file_name = request.form.get("input_file", "")
        filename = None
        if module_name != "" and data != "":
            data = data.splitlines()
            if not isinstance(data, list):
                data = [data]
            input_data = {"text": data}
        # 如果是图片
        elif module_name != "" and img_base64 != "":
            md5_get = hashlib.md5()
            md5_src = str(time.time()) + img_base64
            md5_get.update(md5_src)
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
            results = predict_method(data=input_data)
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
                "output_img": base64_head + "," + output_img_base64
            }
        print(results)
        return {"result": results}

    @app_instance.route("/predict2/<module_name>", methods=["POST"])
    def predict2(module_name):
        if request.method == "GET":
            data = request.args.get("input_text")
            print(data)
            if not isinstance(data, list):
                data = [data]
            input_data = {"text": data}
        elif request.method == "POST":
            data = request.form.get("input_text", "")
            file = request.files.get("input_file", None)
            # 如果是文本
            filename = None
            if module_name != "" and data != "":
                if not isinstance(data, list):
                    data = [data]
                input_data = {"text": data}
            # 如果是文件
            elif file:
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
                file.save(filename)
                filename = [filename]
                if file.mimetype.startswith("image"):
                    input_data = {"image": filename}
                elif file.mimetype.startswith("text"):
                    with open(filename[0], "r") as fp:
                        contents = fp.read()
                        content_lines = contents.splitlines()
                    data = []
                    for sentence in content_lines:
                        if sentence != "":
                            data.append(sentence)
                    print(data)
                    input_data = {"text": data}
                else:
                    return {"result": "不支持的文件类型！"}
        module = TextModelService.get_module(module_name)
        method = module.desc.attr.map.data['default_signature'].s
        if method == "":
            return {"Error": "The module can't be use for predict."}
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            results = predict_method(data=input_data)
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)
        if module.type.startswith("CV"):
            results = str(results)
        if filename is not None:
            os.remove(filename[0])
        print(results)
        return {"result": results}

    @app_instance.route("/inference/<module_name>", methods=["GET", "POST",
                                                             "PUT"])
    def inference(module_name):
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
        module = TextModelService.get_module(module_name)
        method = module.desc.attr.map.data['default_signature'].s
        if method == "":
            return {"Error": "The module can't be use for predict."}
        method_name = module.desc.attr.map.data['default_signature'].s
        if method_name != "":
            predict_method = getattr(module, method_name)
            results = predict_method(data=input_data)
        else:
            results = "您调用的模型{}不支持在线预测".format(module_name)
        if module.type.startswith("CV"):
            results = str(results)
        return {"result": results}

    return app_instance


def run():

    my_app = create_app()
    my_app.run(host="0.0.0.0", port=8888, debug=True)


def test_post(data):
    from serving.model_service.lac_model import lac_module_service
    inputs = {"text": data}
    result = lac_module_service.inference(inputs)
    return result


if __name__ == "__main__":
    # mo = hub.Module("yolov3_coco2017")
    # input_data = {'image': [u'1e647f8c4ef627b41941fd32676105dc.jpg']}
    # print(os.getcwd())
    # results = mo.object_detection(data=input_data)
    # print(results)
    run()
