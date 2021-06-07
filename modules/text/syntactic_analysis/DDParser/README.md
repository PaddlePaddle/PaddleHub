# 命令行预测

```shell
$ hub run ddparser --input_text="百度是一家高科技公司"
```

# API

## parse(texts=[], return\_visual=False)

依存分析接口，输入文本，输出依存关系。

**参数**

* texts(list\[list\[str\] or list\[str\]]): 待预测数据。各元素可以是未分词的字符串，也可以是已分词的token列表。
* return\_visual(bool): 是否返回依存分析可视化结果。如果为True，返回结果中将包含'visual'字段。

**返回**

* results(list\[dict\]): 依存分析结果。每个元素都是dict类型，包含以下信息：  
```python
  {
      'word': list[str], 分词结果。
      'head': list[int], 当前成分其支配者的id。
      'deprel': list[str], 当前成分与支配者的依存关系。
      'prob': list[float], 从属者和支配者依存的概率。
      'postag': list[str], 词性标签，只有当texts的元素是未分词的字符串时包含这个键。
      'visual': 图像数组，可以使用cv2.imshow显示图像或cv2.imwrite保存图像。
  }
```

## visualize(word, head, deprel)

可视化接口，输入依存分析接口得到的信息，输出依存图形数组。

**参数**

* word(list\[list\[str\]\): 分词信息。
* head(list\[int\]): 当前成分其支配者的id。
* deprel(list\[str\]): 当前成分与支配者的依存关系。

**返回**

* data(numpy.array): 图像数组。可以使用cv2.imshow显示图像或cv2.imwrite保存图像。

**代码示例**

```python
import cv2
import paddlehub as hub

module = hub.Module(name="ddparser")

test_text = ["百度是一家高科技公司"]
results = module.parse(texts=test_text)
print(results)

test_tokens = [['百度', '是', '一家', '高科技', '公司']]
results = module.parse(texts=test_text, return_visual = True)
print(results)

result = results[0]
data = module.visualize(result['word'],result['head'],result['deprel'])
# or data = result['visual']
cv2.imwrite('test.jpg',data)
```

# DependencyParser 服务部署

PaddleHub Serving可以部署一个在线情感分析服务，可以将此接口用于在线web应用。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ddparser
```

启动时会显示加载模型过程，启动成功后显示
```shell
Loading ddparser successful.
```

这样就完成了服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

import numpy as np
import cv2

# 待预测数据
text = ["百度是一家高科技公司"]

# 设置运行配置
return_visual = True
data = {"texts": text, "return_visual": return_visual}

# 指定预测方法为DuDepParser并发送post请求，content-type类型应指定json方式
url = "http://0.0.0.0:8866/predict/ddparser"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))
results = r.json()['results']

for i in range(len(results)):
  print(results[i]['word'])
  # 不同于本地调用parse接口，serving返回的图像是list类型的，需要先用numpy加载再显示或保存。
  cv2.imwrite('%s.jpg'%i, np.array(results[i]['visual']))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)


### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0


## 更新历史

* 1.0.0

  初始发布
