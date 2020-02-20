# coding: utf8
import requests
import json
import base64
import os


if __name__ == "__main__":
    file_list = [
        "../../../../docs/imgs/family_mask.jpg",
        "../../../../docs/imgs/woman_mask.jpg"
    ]
    files = [("image", (open(item, "rb"))) for item in file_list]

    # 指定检测方法为pyramidbox_lite_server_mask并发送post请求
    url = "http://127.0.0.1:8866/predict/image/pyramidbox_lite_server_mask"
    r = requests.post(url=url, files=files)

    results = eval(r.json()["results"])

    if not os.path.exists("output"):
        os.mkdir("output")
    for item in results:
        with open(os.path.join("output", item["path"]), "wb") as fp:
            fp.write(base64.b64decode(item["base64"].split(',')[-1]))
            item.pop("base64")

    print(json.dumps(results, indent=4, ensure_ascii=False))
