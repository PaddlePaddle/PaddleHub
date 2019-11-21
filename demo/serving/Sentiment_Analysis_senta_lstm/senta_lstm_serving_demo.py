# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
    text = {"text": text_list}
    # 指定预测方法为senta_lstm并发送post请求
    url = "http://127.0.0.1:8866/predict/text/senta_lstm"
    r = requests.post(url=url, data=text)

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
