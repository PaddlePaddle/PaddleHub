# coding: utf8
from paddlehub.serving.bert_serving import bert_service

if __name__ == "__main__":
    result = bert_service.connect(
        input_text=[
            ["远上寒山石径斜"],
        ],
        model_name="bert_chinese_L-12_H-768_A-12",
        emb_size=768,
        show_ids=True,
        do_lower_case=True,
        port=8866)
    print(result)
