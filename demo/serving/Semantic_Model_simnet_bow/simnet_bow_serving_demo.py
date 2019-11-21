# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定用于用于匹配的文本并生成字典{"text_1": [text_a1, text_a2, ... ]
    #                              "text_2": [text_b1, text_b2, ... ]}
    text = {
        "text_1": ["这道题太难了", "这道题太难了", "这道题太难了"],
        "text_2": ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
    }
    # 指定匹配方法为simnet_bow并发送post请求
    url = "http://127.0.0.1:8866/predict/text/simnet_bow"
    r = requests.post(url=url, data=text)

    # 打印匹配结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
