#coding:utf-8
import paddlehub as hub

senta = hub.Module(name="senta_bilstm")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
input_dict = {"text": test_text}
results = senta.sentiment_classify(data=input_dict)

for result in results:
    print(result['text'])
    print(result['sentiment_label'])
    print(result['sentiment_key'])
    print(result['positive_probs'])
    print(result['negative_probs'])
