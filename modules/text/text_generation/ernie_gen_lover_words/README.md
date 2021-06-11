## 概述

ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。ernie_gen_lover_words采用网络搜集的情诗、情话数据微调，可用于生成情话。
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/resources/multi-flow-attention.png" hspace='10'/> <br />
</p>

更多详情参考论文[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)

## 命令行预测

```shell
$ hub run ernie_gen_lover_words --input_text="情人节" --use_gpu True --beam_width 5
```

## API

```python
def generate(texts, use_gpu=False, beam_width=5):
```

预测API，输入情话开头，输出情话下文。

**参数**

* texts (list\[str\]): 情话的开头；
* use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA\_VISIBLE\_DEVICES环境变量**；
* beam\_width: beam search宽度，决定每个情话开头输出的下文数目。

**返回**

* results (list\[list\]\[str\]): 情话下文，每个情话开头会生成beam_width个下文。

**代码示例**

```python
import paddlehub as hub

module = hub.Module(name="ernie_gen_lover_words")

test_texts = ['情人节', '故乡', '小编带大家了解一下程序员情人节']
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
for result in results:
    print(result)
```

## 服务部署

PaddleHub Serving 可以部署在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ernie_gen_lover_words -p 8866
```

这样就完成了一个服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

### 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

# 发送HTTP请求

data = {'texts':['情人节', '故乡', '小编带大家了解一下程序员情人节'],
        'use_gpu':False, 'beam_width':5}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/ernie_gen_lover_words"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 保存结果
results = r.json()["results"]
for result in results:
    print(result)
```

## 查看代码

https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-gen/

### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0

paddlenlp >= 2.0.0


## 更新历史

* 1.0.0

  初始发布

* 1.0.1

  完善API的输入文本检查

* 1.0.2

  接入PaddleNLP
