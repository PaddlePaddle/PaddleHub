# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text_list = [
        "the plant which is owned by <unk> & <unk> co. was under contract with <unk> to make the cigarette filter",
        "more common <unk> fibers are <unk> and are more easily rejected by the body dr. <unk> explained"
    ]
    text = {"text": text_list}
    # 指定预测方法为lm_lstm并发送post请求
    url = "http://127.0.0.1:8866/predict/text/lm_lstm"
    r = requests.post(url=url, data=text)

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
