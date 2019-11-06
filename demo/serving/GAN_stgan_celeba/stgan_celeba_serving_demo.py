# coding: utf8
import requests
import json
import base64
import os

if __name__ == "__main__":
    file_list = ["../img/woman.png"]
    files = [("image", (open(item, "rb"))) for item in file_list]
    url = "http://127.0.0.1:8866/predict/image/stgan_celeba"

    data = {"info": ["Female,Brown_Hair"], "style": ["Aged"]}
    r = requests.post(url=url, data=data, files=files)

    results = eval(r.json()["results"])
    if not os.path.exists("output"):
        os.mkdir("output")
    for item in results:
        output_path = os.path.join("output", item["path"].split("/")[-1])
        with open(output_path, "wb") as fp:
            fp.write(base64.b64decode(item["base64"].split(',')[-1]))
            item.pop("base64")
    print(json.dumps(results, indent=4, ensure_ascii=False))
