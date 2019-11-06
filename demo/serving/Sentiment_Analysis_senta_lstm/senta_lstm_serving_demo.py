# coding: utf8
import requests
import json

if __name__ == "__main__":
    text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
    text = {"text": text_list}
    url = "http://127.0.0.1:8866/predict/text/senta_lstm"
    r = requests.post(url=url, data=text)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
