# coding: utf8
import requests
import json

if __name__ == "__main__":
    text = {
        "text_1": ["这道题太难了", "这道题太难了", "这道题太难了"],
        "text_2": ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
    }
    url = "http://127.0.0.1:8866/predict/text/simnet_bow"
    r = requests.post(url=url, data=text)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
