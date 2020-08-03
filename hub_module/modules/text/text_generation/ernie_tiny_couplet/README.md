ernie_tiny_couplet是一个对联生成模型，它由ernie_tiny预训练模型经PaddleHub TextGenerationTask微调而来，仅支持预测，如需进一步微调请参考PaddleHub text_generation demo。

```shell
$ hub install ernie_tiny_couplet==1.0.0
```
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/paddlehub-img%2Fernie_tiny_framework.PNG" hspace='10'/> <br />
</p>

本预测module系ernie_tiny预训练模型经由TextGenerationTask微调而来，有关ernie\_tiny的介绍请参考[ernie_tiny module](https://www.paddlepaddle.org.cn/hubdetail?name=ernie_tiny&en_category=SemanticModel)，微调方式请参考[text_generation demo](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.8/demo/text_generation)，预训练模型转换成预测module的转换方式请参考[Fine-tune保存的模型如何转化为一个PaddleHub Module](https://github.com/PaddlePaddle/PaddleHub/blob/develop/docs/tutorial/finetuned_model_to_module.md)

## 命令行预测

```shell
$ hub run ernie_tiny_couplet --input_text '风吹云乱天垂泪'
```
命令行预测只支持使用CPU预测，如需使用GPU，请使用API方式预测。

## API
```python
def generate(texts)
```

对联预测接口，输入上联文本，输出下联文本。该接口封装了上联文本使用`hub.BertTokenizer`编码的过程，因此它的调用方式比demo中提供的[predcit接口](https://github.com/PaddlePaddle/PaddleHub/blob/develop/demo/text_generation/predict.py#L83)简单。

**参数**

> texts(list\[str\])： 上联文本。

**返回**

> result(list\[str\]): 下联文本。每个上联会对应输出10个下联。

**代码示例**

```python
import paddlehub as hub

# Load ernie pretrained model
module = hub.Module(name="ernie_tiny_couplet", use_gpu=True)
results = module.generate(["风吹云乱天垂泪", "若有经心风过耳"])
for result in results:
    print(result)
```

## 服务部署

PaddleHub Serving 可以部署在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ernie_tiny_couplet
```

这样就完成了一个服务化API的部署，默认端口号为8866。

**NOTE:** 服务部署只支持使用CPU，如需使用GPU，请使用API方式预测。

### 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

# 发送HTTP请求

data = {'texts':["风吹云乱天垂泪", "若有经心风过耳"]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/ernie_tiny_couplet"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 保存结果
results = r.json()["results"]
print(results)
```

##   查看代码

https://github.com/PaddlePaddle/PaddleHub/blob/develop/demo/text_generation


## 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.8.0

## 更新历史

* 1.0.0

  初始发布。
