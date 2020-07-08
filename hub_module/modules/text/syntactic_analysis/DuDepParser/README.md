# 命令行预测

```shell
$ hub run DuDepParser --input_text="百度是一家高科技公司"
```

# API

## parse(texts=[], use_gpu=False, batch_size=1)

依存分析接口，输入文本，输出依存关系。

**参数**

* texts(list): 待预测数据
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小

**返回**

* results(list): 依存分析结果

# DependencyParser 服务部署

PaddleHub Serving可以部署一个在线情感分析服务，可以将此接口用于在线web应用。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m DuDepParser
```

启动时会显示加载模型过程，启动成功后显示
```shell
Loading DuDepParser successful.
```

这样就完成了服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

# 待预测数据
text = ["百度是一家高科技公司"]

# 设置运行配置
# 对应本地预测DuDepParser.parse(texts=text, batch_size=1, use_gpu=False)
data = {"texts": text, "batch_size": 1, "use_gpu":False}

# 指定预测方法为DuDepParser并发送post请求，content-type类型应指定json方式
url = "http://0.0.0.0:8866/predict/DuDepParser"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)


### 依赖

paddlepaddle >= 1.8.2

paddlehub >= 1.7.0

LAC >= 0.1.4

python >= 3.7.0


## 更新历史

* 1.0.0

  初始发布
