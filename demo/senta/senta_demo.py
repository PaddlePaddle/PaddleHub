#coding:utf-8
from __future__ import print_function

import json
import os
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load Senta-BiLSTM module
    senta = hub.Module(name="senta_bilstm")

    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    input_dict = {"text": test_text}

    results = senta.sentiment_classify(data=input_dict)

    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(
                json.dumps(results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
