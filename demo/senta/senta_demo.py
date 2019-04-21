# coding: utf-8
import os
import paddlehub as hub

if __name__ == "__main__":
    # Load Senta-BiLSTM module
    senta = hub.Module(name="senta")

    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    input_dict = {"text": test_text}

    results = senta.sentiment_classify(data=input_dict)
    for index, result in enumerate(results):
        print(test_text[index], result['sentiment_key'])
