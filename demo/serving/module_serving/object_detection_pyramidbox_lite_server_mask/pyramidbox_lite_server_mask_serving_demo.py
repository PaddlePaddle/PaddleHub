# coding: utf8
import requests
import json
import base64


if __name__ == "__main__":
    # 获取第一张图片文件的base64编码
    with open(file="../../../../docs/imgs/family_mask.jpg", mode="rb") as fp:
        base64_data1 = base64.b64encode(fp.read())
    base64_data1 = str(base64_data1, encoding="utf8")
    # 获取第二张图片文件的base64编码
    with open(file="../../../../docs/imgs/woman_mask.jpg", mode="rb") as fp:
        base64_data2 = base64.b64encode(fp.read())
    base64_data2 = str(base64_data2, encoding="utf8")

    data = {"b64s": [base64_data1, base64_data2]}
    data = {"data": json.dumps(data)}

    # 指定检测方法为pyramidbox_lite_server_mask并发送post请求
    url = "http://127.0.0.1:8866/predict/image/pyramidbox_lite_server_mask"
    r = requests.post(url=url, data=data)

    # 得到并打印检测结果
    results = eval(r.json()["results"])

    print(json.dumps(results, indent=4, ensure_ascii=False))
