# coding: utf8
import requests
import json
import base64
import os

if __name__ == "__main__":
    # 指定要使用的图片文件并生成列表[("image", img_1), ("image", img_2), ... ]
    file_list = ["../img/girl.jpg"]
    files = [("image", (open(item, "rb"))) for item in file_list]
    # 指定图片分割方法为deeplabv3p_xception65_humanseg并发送post请求
    url = "http://127.0.0.1:8866/predict/image/deeplabv3p_xception65_humanseg"
    r = requests.post(url=url, files=files)

    results = eval(r.json()["results"])

    # 保存分割后的图片到output文件夹，打印模型输出结果
    if not os.path.exists("output"):
        os.mkdir("output")
    for item in results:
        with open(
                os.path.join("output", item["processed"].split("/")[-1]),
                "wb") as fp:
            fp.write(base64.b64decode(item["base64"].split(',')[-1]))
            item.pop("base64")
    print(json.dumps(results, indent=4, ensure_ascii=False))
