# coding: utf8
import requests
import json

if __name__ == "__main__":
    text_list = ["今天是个好日子", "天气预报说今天要下雨"]
    text = {"text": text_list}
    # 将用户自定义词典文件发送到预测接口即可
    file = {"user_dict": open("dict.txt", "rb")}
    url = "http://127.0.0.1:8866/predict/text/lac"
    r = requests.post(url=url, files=file, data=text)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
