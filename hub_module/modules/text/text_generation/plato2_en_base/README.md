## 概述

PLATO2是一个超大规模生成式对话系统模型。它承袭了PLATO隐变量进行回复多样化生成的特性，能够就开放域话题进行流畅深入的聊天。据公开数据，其效果超越了Google 于2020年2月份发布的 Meena和Facebook AI Research于2020年4月份发布的Blender的效果。plato2_en_base包含310M参数，可用于一键预测对话回复，该Module仅支持使用GPU预测，不支持CPU。
<p align="center">
<img src="https://image.jiqizhixin.com/uploads/editor/65107b78-0259-4121-b8c5-a090f9d3175b/640.png" hspace='10'/> <br />
</p>

更多详情参考论文[PLATO-2: Towards Building an Open-Domain Chatbot via Curriculum Learning](https://arxiv.org/abs/2006.16779)

## 命令行预测

```shell
$ hub run plato2_en_base --input_text="Hello, how are you" --use_gpu
```

## API

```python
def generate(texts):
```

预测API，输入对话上下文，输出机器回复。

**参数**

* texts (list\[str\]): 每个元素为一次对话的上下文，上下文应包含人类和机器人的对话内容，不同角色之间的聊天用分隔符"\t"进行分割；例如[["Hello\thi, nice to meet you\tnice to meet you"]]。这个输入中包含1次对话，机器人回复了"hi, nice to meet you"后人类回复“nice to meet you”，现在轮到机器人回复了。

**返回**

* results (list\[str\]): 每个元素为相应对话中机器人的新回复。

**代码示例**

```python
import paddlehub as hub

module = hub.Module(name="plato2_en_base")

test_texts = ["Hello","Hello\thi, nice to meet you\tnice to meet you"]
results = module.generate(texts=test_texts)
for result in results:
    print(result)
```

```python
def interact():
```

进入交互模式，交互模式中，控制台将持续监听键盘输入，机器人将即时回复消息。

**代码示例**

```python
import paddlehub as hub

module = hub.Module(name="plato2_en_base")

module.interact()
```

## 服务部署

PaddleHub Serving 可以部署在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m plato2_en_base -p 8866
```

这样就完成了一个服务化API的部署，默认端口号为8866。

**NOTE:** 在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量。

### 第二步：发送预测请求

方式1： 自定义脚本发送对话信息

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

# 发送HTTP请求

data = {'texts':["Hello","Hello\thi, nice to meet you\tnice to meet you"]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/plato2_en_base"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 保存结果
results = r.json()["results"]
for result in results:
    print(result)
```

方式2： 通过交互式客户端进入交互模式

下载交互式客户端：

```
wget https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.8/hub_module/modules/text/text_generation/plato2_en_base/plato2_en_base_client.py
```

进入交互模式：
```
python plato2_en_base_client.py
```
默认地址为127.0.0.1，您可以通过 -a 进行设置。默认端口为8866，您可以通过 -p 进行设置。

## 查看代码

https://github.com/PaddlePaddle/Knover

### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0


## 更新历史

* 1.0.0

  初始发布
