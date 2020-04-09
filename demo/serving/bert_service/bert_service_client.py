# coding: utf8
from paddlehub.serving.bert_serving import bs_client

if __name__ == "__main__":
    # 初始化bert_service客户端BSClient
    bc = bs_client.BSClient(module_name="ernie_tiny", server="127.0.0.1:8866")

    # 输入要做embedding的文本
    # 文本格式为[["文本1"], ["文本2"], ]
    input_text = [
        ["西风吹老洞庭波"],
        ["一夜湘君白发多"],
        ["醉后不知天在水"],
        ["满船清梦压星河"],
    ]

    # BSClient.get_result()获取结果
    result = bc.get_result(input_text=input_text)

    # 打印输入文本的embedding结果
    for item in result:
        print(item)
