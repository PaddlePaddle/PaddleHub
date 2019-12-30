# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定要预测的图片并生成列表[("image", img_1), ("image", img_2), ... ]
    file_list = ["../img/cat.jpg", "../img/flower.jpg"]
    files = [("image", (open(item, "rb"))) for item in file_list]
    # 指定预测方法为vgg11_imagenet并发送post请求
    url = "http://127.0.0.1:8866/predict/image/vgg11_imagenet"
    r = requests.post(url=url, files=files)

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
