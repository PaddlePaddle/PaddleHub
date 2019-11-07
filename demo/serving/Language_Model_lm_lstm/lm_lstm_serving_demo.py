# coding: utf8
import requests
import json

if __name__ == "__main__":
    text_list = [
        "the plant which is owned by <unk> & <unk> co. was under contract with <unk> to make the cigarette filter",
        "more common <unk> fibers are <unk> and are more easily rejected by the body dr. <unk> explained"
    ]
    text = {"text": text_list}
    url = "http://127.0.0.1:8866/predict/text/lm_lstm"
    r = requests.post(url=url, data=text)

    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
