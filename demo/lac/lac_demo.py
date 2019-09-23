#coding:utf-8
from __future__ import print_function

import json
import os
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load LAC Module
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]

    # Set input dict
    inputs = {"text": test_text}

    # execute predict and print the result
    results = lac.lexical_analysis(data=inputs, use_gpu=True, batch_size=10)
    for result in results:
        if six.PY2:
            print(
                json.dumps(result['word'], encoding="utf8", ensure_ascii=False))
            print(
                json.dumps(result['tag'], encoding="utf8", ensure_ascii=False))
        else:
            print(result['word'])
            print(result['tag'])
