# coding: utf-8
from flask import Flask, request, render_template
from paddlehub.serving.model_service.text_model_service import TextModelService
from paddlehub.serving.model_service.image_model_service import ImageModelService
from paddlehub.serving import utils
# from model_service.text_model_service import TextModelService
# from model_service.image_model_service import ImageModelService
import json
import hashlib
import time
import os
import base64
import logging
import cv2
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import random
from Queue import PriorityQueue


class MyPriorityQueue(PriorityQueue):
    def get_attribute(self, name):
        return getattr(self, name)


class Manager(BaseManager):
    pass


Manager.register("get_priorityQueue", MyPriorityQueue)
m = Manager()
m.start()

use_gpu = False
nlp_module = ["lac", "senta_lstm"]
cv_module = ["yolov3_coco2017", "faster_rcnn_coco2017"]

lac_queue = m.get_priorityQueue(maxsize=20)
lac_batch_size = 3
lac_name = "lac"
yolov3_coco2017_queue = m.get_priorityQueue(maxsize=20)
yolov3_coco2017_batch_size = 3
yolov3_coco2017_name = "yolov3_coco2017"
faster_rcnn_coco2017_queue = m.get_priorityQueue(maxsize=20)
faster_rcnn_coco2017_batch_size = 3
faster_rcnn_coco2017_name = "faster_rcnn_coco2017"
senta_lstm_queue = m.get_priorityQueue(maxsize=20)
senta_lstm_batch_size = 3
senta_lstm_batch_name = "senta_lstm_batch"
queues_list = [
    lac_queue, yolov3_coco2017_queue, senta_lstm_queue,
    faster_rcnn_coco2017_queue
]
batch_size_list = [
    lac_batch_size, yolov3_coco2017_batch_size, senta_lstm_batch_size,
    faster_rcnn_coco2017_batch_size
]
name_list = [
    lac_name, yolov3_coco2017_name, senta_lstm_batch_name,
    faster_rcnn_coco2017_name
]
queues_dict = {
    "lac": lac_queue,
    "yolov3_coco2017": yolov3_coco2017_queue,
    "senta_lstm": senta_lstm_queue,
    "faster_rcnn_coco2017": faster_rcnn_coco2017_queue
}
queue_name_list = [
    "lac", "yolov3_coco2017", "senta_lstm", "faster_rcnn_coco2017"
]
results_dict = mp.Manager().dict()


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
            results = predict_method(data=real_input_data, use_gpu=use_gpu)
        except Exception as err:
            return {"result": "请检查数据格式!"}
    else:
        results = "您调用的模型{}不支持在线预测".format(module_name)
    try:
        result_data = []
        for index in range(len(input_data)):
            result_data.append(list(input_data[index]))
            result_data[-1][3] = results[index]
    except Exception as err:
        print("转换出现问题")
    # 把预测结果存在全局hash表中，按照req_id划分，每个value是一个优先级队列
    # 查看是否已经存在req_id对应的部分结果
    for index in range(len(result_data)):
        if results_dict.get(result_data[index][2]) is None:
            results_dict[result_data[index][2]] = [[
                result_data[index][1], result_data[index][3]
            ]]
        else:
            temp_list = results_dict[result_data[index][2]]
            temp_list.append([result_data[index][1], result_data[index][3]])
            results_dict[result_data[index][2]] = temp_list
    # 结果已经存储，主进程取轮询查找即可
    return {"result": results_dict}


# 这里的input_data应当是一个文件名列表
def predict_cv(input_data, module_name, batch_size=1):
    global use_gpu
    filename_list = []
    for index in range(len(input_data)):
        filename_list.append(input_data[index][3])
        cv2.imread(input_data[index][3])
    input_images = {"image": filename_list}
    module = ImageModelService.get_module(module_name)
    method_name = module.desc.attr.map.data['default_signature'].s
    if method_name != "":
        predict_method = getattr(module, method_name)
        try:
            results = predict_method(
                data={"image": filename_list}, use_gpu=use_gpu)
        except Exception as err:
            return {"result": "请检查数据格式!"}
    else:
        results = "您调用的模型{}不支持在线预测".format(module_name)
    # 这里已经生成了多个结果在output文件夹中，只要把这个路径读出来放回input_data中就可以了
    try:
        result_data = []
        for index in range(len(input_data)):
            result_data.append(list(input_data[index]))
            result_data[-1][3] = results[index]
    except Exception as err:
        print("转换出现问题")
    # 把预测结果存在全局hash表中，按照req_id划分，每个value是一个优先级队列
    # 查看是否已经存在req_id对应的部分结果
    for index in range(len(result_data)):
        if results_dict.get(result_data[index][2]) is None:
            results_dict[result_data[index][2]] = [[
                result_data[index][1], result_data[index][3]
            ]]
        else:
            temp_list = results_dict[result_data[index][2]]
            temp_list.append([result_data[index][1], result_data[index][3]])
            results_dict[result_data[index][2]] = temp_list
    # 结果已经存储，主进程取轮询查找即可
    return {"result": results}


def worker():
    global batch_size_list, name_list, queue_name_list, cv_module
    latest_num = random.randrange(0, len(queue_name_list))
    while True:
        time.sleep(0.000001)
        for index in range(len(queue_name_list)):
            while queues_dict[queue_name_list[latest_num]].empty() is not True:
                input_data = []
                for index2 in range(batch_size_list[latest_num]):
                    if queues_dict[queue_name_list[latest_num]].empty() is True:
                        break
                    input_data.append(
                        queues_dict[queue_name_list[latest_num]].get())
                # 取出了数据，现在进行实际预测，先用lac进行测试
                if len(input_data) != 0:
                    choose_module_category(input_data,
                                           queue_name_list[latest_num],
                                           batch_size_list[latest_num])
                else:
                    pass
                # 预测完毕的结果存进去，又可以继续了
            latest_num = (latest_num + 1) % len(queue_name_list)


def create_app():
    # app_instance = Flask("paddlehub.serving.app")
    app_instance = Flask(__name__)
    app_instance.config["JSON_AS_ASCII"] = False
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app_instance.logger.handlers = gunicorn_logger.handlers
    app_instance.logger.setLevel(gunicorn_logger.level)
    global queues_dict
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    for i in range(mp.cpu_count() - 1):
        pool.apply_async(worker)
    # pool = mp.Pool(processes=(mp.cpu_count()-1))
    # for i in range((mp.cpu_count()-1)):
    #     pool.apply_async(worker)

    @app_instance.route("/", methods=["GET", "POST"])
    def index():
        return render_template("main.html")

    @app_instance.before_request
    def before_request():
        # print("start, time= ", time.time())
        # # global waiting_queue
        # print(request.url)
        # print(request.path)
        # data = "data"
        # try:
        #     waiting_queue.put(data, block=False)
        # except Queue.Full:
        #     return {"result": "现在使用serving的人太多了，请稍后再尝试，谢谢！🙏"}
        # print(request.form)
        # print(request.data)
        # print(request.path)
        # print(request.url)
        # print(request.form)
        request.data = {"id": str(time.time())}
        pass

    # @app_instance.route("/predict/image/<module_name>", methods=["POST"])
    # def predict_iamge(module_name):
    #     global use_gpu
    #     img_base64 = request.form.get("input_img", "")
    #     received_file_name = request.form.get("input_file", "")
    #     md5_get = hashlib.md5()
    #     md5_src = str(time.time()) + str(img_base64)
    #     md5_get.update(md5_src.encode("utf-8"))
    #
    #     ext = received_file_name.split(".")[-1]
    #     if ext == "":
    #         return {"result": "未识别的文件类型"}
    #     filename = md5_get.hexdigest() + "." + ext
    #     base64_head = img_base64.split(',')[0]
    #     img_data = base64.b64decode(img_base64.split(',')[-1])
    #     with open(filename, "wb") as fp:
    #         fp.write(img_data)
    #     input_data = {"image": [filename]}
    #
    #     module = ImageModelService.get_module(module_name)
    #     method_name = module.desc.attr.map.data['default_signature'].s
    #     if method_name != "":
    #         predict_method = getattr(module, method_name)
    #         print(input_data)
    #         try:
    #             print("Use gpu is", use_gpu)
    #             results = predict_method(data=input_data, use_gpu=use_gpu)
    #         except Exception as err:
    #             print(err)
    #             return {"result": "请检查数据格式!"}
    #     else:
    #         results = "您调用的模型{}不支持在线预测".format(module_name)
    #     os.remove(filename)
    #     output_file = os.path.join("./output", filename)
    #     if output_file is not None and os.path.exists(output_file):
    #         with open(output_file, "rb") as fp:
    #             output_img_base64 = base64.b64encode(fp.read())
    #         os.remove(output_file)
    #     if module.type.startswith("CV"):
    #         results = {
    #             "border": str(results[0]["data"]),
    #             "output_img": base64_head + ","
    #                           + str(output_img_base64).replace("b'", "").replace("'", "")
    #         }
    #     return {"result": results}

    @app_instance.route("/predict/image/<module_name>", methods=["POST"])
    def predict_iamge(module_name):
        # module_name = "faster_rcnn_coco2017"
        global results_dict
        req_id = request.data.get("id")
        img_base64 = request.form.get("input_img", "")
        received_file_name = request.form.get("input_file", "")
        ext = received_file_name.split(".")[-1]
        if ext == "":
            return {"result": "未识别的文件类型"}
        score = time.time()
        filename = utils.gen_md5(str(time.time()) + str(img_base64)) + "." + ext
        base64_head = img_base64.split(',')[0]
        img_data = base64.b64decode(img_base64.split(',')[-1])
        with open(filename, "wb") as fp:
            fp.write(img_data)
        # input_data = {"image": [filename]}
        file_list = [filename]
        # 放入
        if queues_dict[module_name].qsize(
        ) + 1 > queues_dict[module_name].get_attribute("maxsize"):
            return {"result": "当前访问人数过多，请稍后再来"}
        data_2_item(file_list, req_id, score, module_name)
        data_num = len(file_list)
        # 轮询查找结果
        results = []
        # 开始轮询查找
        result_len = 0
        while result_len != data_num:
            # time.sleep(5)
            result_len = len(results_dict.get(req_id, []))
        results = results_dict.get(req_id)
        return {"result": str(results)}

    def data_2_item(data_list, req_id, score, module_name):
        global queues_dict
        item_list = []
        # print("开始放入")
        # print("放%s个" % len(data_list))
        for index in range(len(data_list)):
            queues_dict[module_name].put((score, index, req_id,
                                          data_list[index]))

    @app_instance.route("/down", methods=["GET", "POST"])
    def down():
        strr = ""
        for key, value in queues_dict.items():
            strr += key
            strr += " : "
            while value.qsize() != 0:
                strr += str(value.get())
                strr += ","
        return {"r": strr}

    @app_instance.route("/predict/text/<module_name>", methods=["POST"])
    def predict_text(module_name):
        global results_dict, queues_dict
        req_id = request.data.get("id")
        data_list = request.form.get("input_text")
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
            return {"result": ["请求数量过多，请减少一次请求数量"]}
        if data_num + queues_dict[module_name].qsize(
        ) > queues_dict[module_name].get_attribute("maxsize"):
            return {"result": "当前访问人数过多，请稍后再来"}
        data_2_item(data_list, req_id, score, module_name)
        results = []
        # 开始轮询查找
        result_len = 0
        while result_len != data_num:
            # time.sleep(5)
            result_len = len(results_dict.get(req_id, []))
        results = results_dict.get(req_id)
        results = [i[1] for i in sorted(results, key=lambda k: k[0])]
        return {"result": results}

    # 此方法为废弃方法
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
                "border":
                str(results[0]["data"]),
                "output_img":
                base64_head + "," + str(output_img_base64).replace(
                    "b'", "").replace("'", "")
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
