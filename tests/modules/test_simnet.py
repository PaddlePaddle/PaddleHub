#coding:utf-8
import paddlehub as hub

simnet_bow = hub.Module(name="simnet_bow")
test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

inputs = {"text_1": test_text_1, "text_2": test_text_2}
results = simnet_bow.similarity(data=inputs)

max_score = -1
result_text = ""
for result in results:
    if result['similarity'] > max_score:
        max_score = result['similarity']
        result_text = result['text_2']

print("The most matching with the %s is %s" % (test_text_1[0], result_text))
