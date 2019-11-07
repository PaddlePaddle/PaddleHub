# coding: utf8
import requests
import json

if __name__ == "__main__":
    text_list = ["今天是个好日子", "天气预报说今天要下雨"]
    text = {"text": text_list}
    url = "http://127.0.0.1:8866/predict/text/lac"
    r = requests.post(url=url, data=text)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
