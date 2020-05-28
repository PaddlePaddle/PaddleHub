## 概述

近日，百度正式发布情感预训练模型SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）。SKEP利用情感知识增强预训练模型， 在14项中英情感分析典型任务上全面超越SOTA，此工作已经被ACL 2020录用。

SKEP（Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis）是百度研究团队提出的基于情感知识增强的情感预训练算法，此算法采用无监督方法自动挖掘情感知识，然后利用情感知识构建预训练目标，从而让机器学会理解情感语义，在14项中英情感分析典型任务上全面超越SOTA。SKEP为各类情感分析任务提供统一且强大的情感语义表示。ernie_skep_sentiment_analysis Module可用于句子级情感分析任务预测。其在预训练时使用ERNIE 1.0 large预训练参数作为其网络参数初始化继续预训练。同时，该Module支持完成句子级情感分析任务迁移学习Fine-tune。

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/model/nlp/skep.png" hspace='10'/> <br />
</p>

更多详情参考ACL 2020论文[SKEP: Sentiment Knowledge Enhanced Pre-training for Sentiment Analysis](https://arxiv.org/abs/2005.05635)

## 命令行预测

```shell
$ hub run ernie_skep_sentiment_analysis --input_text='虽然小明很努力，但是他还是没有考100分'
```

## API

```python
def classify_sentiment(texts=[], use_gpu=False)
```

预测API，分类输入文本的情感极性。

**参数**

* texts (list\[str\]): 待预测文本；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**；

**返回**

* res (list\[dict\]): 情感分类结果的列表，列表中每一个元素为 dict，各字段为：
    * text(str): 输入预测文本
    * sentiment_label(str): 情感分类结果，或为positive或为negative
    * positive_probs: 输入预测文本情感极性属于positive的概率
    * negative_probs: 输入预测文本情感极性属于negative的概率

```python
def context(trainable=True, max_seq_len=128)
```
用于获取Module的上下文信息，得到输入、输出以及预训练的Paddle Program副本

**参数**
* trainable(bool): 设置为True时，Module中的参数在Fine-tune时也会随之训练，否则保持不变。
* max_seq_len(int): SKEP模型的最大序列长度，若序列长度不足，会通过padding方式补到**max_seq_len**, 若序列长度大于该值，则会以截断方式让序列长度为**max_seq_len**，max_seq_len可取值范围为0～512；

**返回**
* inputs: dict类型，各字段为：
  * input_ids(Variable): Token Embedding，shape为\[batch_size, max_seq_len\]，dtype为int64类型；
  * position_id(Variable): Position Embedding，shape为\[batch_size, max_seq_len\]，dtype为int64类型；
  * segment_ids(Variable): Sentence Embedding，shape为\[batch_size, max_seq_len\]，dtype为int64类型；
  * input_mask(Variable): token是否为padding的标识，shape为\[batch_size, max_seq_len\]，dtype为int64类型；

* outputs：dict类型，Module的输出特征，各字段为：
  * pooled_output(Variable): 句子粒度的特征，可用于文本分类等任务，shape为 \[batch_size, 768\]，dtype为int64类型；
  * sequence_output(Variable): 字粒度的特征，可用于序列标注等任务，shape为 \[batch_size, seq_len, 768\]，dtype为int64类型；

* program：包含该Module计算图的Program。

```python
def get_embedding(texts, use_gpu=False, batch_size=1)
```

用于获取输入文本的句子粒度特征与字粒度特征

**参数**

* texts(list)：输入文本列表，格式为\[\[sample\_a\_text\_a, sample\_a\_text\_b\], \[sample\_b\_text\_a, sample\_b\_text\_b\],…,\]，其中每个元素都是一个样例，每个样例可以包含text\_a与text\_b。
* use_gpu(bool)：是否使用gpu，默认为False。对于GPU用户，建议开启use_gpu。**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**；

**返回**

* results(list): embedding特征，格式为\[\[sample\_a\_pooled\_feature, sample\_a\_seq\_feature\], \[sample\_b\_pooled\_feature, sample\_b\_seq\_feature\],…,\]，其中每个元素都是对应样例的特征输出，每个样例都有句子粒度特征pooled\_feature与字粒度特征seq\_feature。

```python
def get_params_layer()
```

用于获取参数层信息，该方法与ULMFiTStrategy联用可以严格按照层数设置分层学习率与逐层解冻。

**参数**

* 无

**返回**

* params_layer(dict): key为参数名，值为参数所在层数

**代码示例**

```python
import paddlehub as hub

# Load ernie_skep_sentiment_analysis module.
module = hub.Module(name="ernie_skep_sentiment_analysis")

# Predict sentiment label
test_texts = ['你不是不聪明，而是不认真', '虽然小明很努力，但是他还是没有考100分']
results = module.classify_sentiment(test_texts, use_gpu=False)

# Get feature and main program of ernie_skep_sentiment_analysis
inputs, outputs, program = module.context(trainable=True, max_seq_len=128)

# Must feed all the tensor of ernie_skep_sentiment_analysis's module need
input_ids = inputs["input_ids"]
position_ids = inputs["position_ids"]
segment_ids = inputs["segment_ids"]
input_mask = inputs["input_mask"]

# Use "pooled_output" for sentence-level output.
pooled_output = outputs["pooled_output"]

# Use "sequence_output" for token-level output.
sequence_output = outputs["sequence_output"]

# Get embedding feature.
embedding_result = module.get_embedding(texts=[["Sample1_text_a"],["Sample2_text_a","Sample2_text_b"]], use_gpu=True)

# Get params layer for ULMFiTStrategy.
params_layer = module.get_params_layer()
strategy = hub.finetune.strategy.ULMFiTStrategy(frz_params_layer=params_layer, dis_params_layer=params_layer)
```
利用该PaddleHub Module Fine-tune示例，可参考[文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7.0/demo/text-classification)。

**Note**：建议该PaddleHub Module在**GPU**环境中运行。如出现显存不足，可以将**batch_size**或**max_seq_len**调小。  

## 服务部署

PaddleHub Serving 可以部署一个目标检测的在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ernie_skep_sentiment_analysis
```

这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

### 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'texts':['你不是不聪明，而是不认真', '虽然小明很努力，但是他还是没有考100分']}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/ernie_skep_sentiment_analysis"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

## 查看代码

https://github.com/baidu/Senta

### 依赖

paddlepaddle >= 1.8.0

paddlehub >= 1.7.0


## 更新历史

* 1.0.0

  初始发布
