```shell
$ hub install unified_transformer_12L_cn==1.0.0
```

## 概述

近年来，人机对话系统受到了学术界和产业界的广泛关注并取得了不错的发展。开放域对话系统旨在建立一个开放域的多轮对话系统，使得机器可以流畅自然地与人进行语言交互，既可以进行日常问候类的闲聊，又可以完成特定功能，以使得开放域对话系统具有实际应用价值。具体的说，开放域对话可以继续拆分为支持不同功能的对话形式，例如对话式推荐，知识对话技术等，如何解决并有效融合以上多个技能面临诸多挑战。

[UnifiedTransformer](https://arxiv.org/abs/2006.16779)以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，采用灵活的注意力机制，十分适合文本生成任务，并在模型输入中加入了标识不同对话技能的special token，使得模型能同时支持闲聊对话、推荐对话和知识对话。

unified_transformer_12L_cn包含12层的transformer结构，头数为12，隐藏层参数为768，参数量为132M。该预训练模型使用了样本量为60M的文本数据和20M的对话数据的大型中文对话数据集进行预训练，具体训练详情可以查看[LUGE-Dialogue](https://github.com/PaddlePaddle/Knover/tree/luge-dialogue/luge-dialogue)。

## API

```python
def predict(data: Union[List[List[str]], str],
            max_seq_len: int = 512,
            batch_size: int = 1,
            use_gpu: bool = False,
            **kwargs):
```
预测API，输入对话上下文，输出机器回复。

**参数**
- `data`(Union[List[List[str]], str]): 在非交互模式中，数据类型为List[List[str]]，每个样本是一个List[str]，表示为对话内容
- `max_seq_len`(int): 每个样本的最大文本长度
- `batch_size`(int): 进行预测的batch_size
- `use_gpu`(bool): 是否使用gpu执行预测
- `kwargs`: 预测时传给模型的额外参数，以keyword方式传递。其余的参数详情请查看[UnifiedTransformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer)。

**返回**
* `results`(List[str]): 每个元素为相应对话中模型的新回复

```python
def interactive_mode(max_turn=3)
```
进入交互模式。交互模式中，predict接口的data将支持字符串类型。

**参数**
- `max_turn`(int): 模型能记忆的对话轮次，当`max_turn`为1时，模型只能记住当前对话，无法获知之前的对话内容。


**代码示例**

```python
# 非交互模式
import paddlehub as hub

model = hub.Module(name='unified_transformer_12L_cn')
data = [["你是谁？"], ["你好啊。", "吃饭了吗？",]]
result = model.predict(data)
```

```python
# 交互模式
import paddlehub as hub

model = hub.Module(name='unified_transformer_12L_cn')
with model.interactive_mode(max_turn=3):
    while True:
        human_utterance = input("[Human]: ").strip()
        robot_utterance = model.predict(human_utterance)[0]
        print("[Bot]: %s"%robot_utterance)
```

## 服务部署

PaddleHub Serving可以部署在线服务。

### Step1: 启动PaddleHub Serving

运行启动命令：

```shell
$ hub serving start -m unified_transformer_12L_cn
```

这样就完成了一个对话机器人服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

### Step2: 发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

texts = [["今天是个好日子"], ["天气预报说今天要下雨"]]
data = {"data": texts}
# 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
url = "http://127.0.0.1:8866/predict/unified_transformer_12L_cn"
# 指定post请求的headers为application/json方式
headers = {"Content-Type": "application/json"}

r = requests.post(url=url, headers=headers, data=json.dumps(data))
print(r.json())
```

## 查看代码

https://github.com/PaddlePaddle/Knover

## 依赖

paddlepaddle >= 2.0.0

paddlehub >= 2.1.0

## 更新历史

* 1.0.0

  初始发布
