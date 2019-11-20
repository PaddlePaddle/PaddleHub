# coding: utf8
from paddlehub.serving.bert_serving import bert_service

if __name__ == "__main__":
    # 输入要做embedding的文本
    # 文本格式为[["文本1"], ["文本2"], ]
    input_text = [
        ["西风吹老洞庭波"],
        ["一夜湘君白发多"],
        ["醉后不知天在水"],
        ["满船清梦压星河"],
    ]
    # 调用客户端接口bert_service.connect()获取结果
    result = bert_service.connect(
        input_text=input_text,
        model_name="bert_chinese_L-12_H-768_A-12",
        server="127.0.0.1:8866")

    # 打印embedding结果
    for item in result:
        print(item)
