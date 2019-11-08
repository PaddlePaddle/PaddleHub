# coding: utf8
import requests
import json

if __name__ == "__main__":
    file_list = ["../img/cat.jpg", "../img/flower.jpg"]
    files = [("image", (open(item, "rb"))) for item in file_list]
    url = "http://127.0.0.1:8866/predict/image/vgg11_imagenet"
    r = requests.post(url=url, files=files)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
