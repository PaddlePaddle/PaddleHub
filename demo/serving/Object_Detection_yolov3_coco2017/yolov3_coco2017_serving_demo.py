# coding: utf8
import requests
import json
import base64
import os

if __name__ == "__main__":
    # 指定要检测的图片并生成列表[("image", img_1), ("image", img_2), ... ]
    file_list = ["../img/cat.jpg", "../img/dog.jpg"]
    files = [("image", (open(item, "rb"))) for item in file_list]
    # 指定检测方法为yolov3_coco2017并发送post请求
    url = "http://127.0.0.1:8866/predict/image/yolov3_coco2017"
    r = requests.post(url=url, files=files)

    results = eval(r.json()["results"])

    # 保存检测生成的图片到output文件夹，打印模型输出结果
    if not os.path.exists("output"):
        os.mkdir("output")
    for item in results:
        with open(os.path.join("output", item["path"]), "wb") as fp:
            fp.write(base64.b64decode(item["base64"].split(',')[-1]))
            item.pop("base64")
    print(json.dumps(results, indent=4, ensure_ascii=False))
