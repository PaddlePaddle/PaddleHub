# SimnetBOW API说明

## similarity(texts=[], data={}, use_gpu=False, batch_size=1)

simnet_bow预测接口，计算两个句子的cosin相似度

**参数**

* texts(list): 待预测数据，第一个元素(list)为第一顺序句子，第二个元素(list)为第二顺序句子，两个元素长度相同。
如texts=[["这道题太难了", "这道题太难了", "这道题太难了"], ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]]。
如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为'text_1' 和'text_2'，相应的value(list)是第一顺序句子和第二顺序句子。
如data={"text_1": ["这道题太难了", "这道题太难了", "这道题太难了"], "text_2": ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]}。
如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* data(dict): 预测数据，key必须为'text_1' 和'text_2'，相应的value(list)是第一顺序句子和第二顺序句子。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小

**返回**

* results(list): 带预测数据的cosin相似度

## context(trainable=False)

获取simnet_bow的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* program(Program): 带有预训练参数的program

## get_vocab_path()

获取预训练时使用的词汇表

**返回**

* vocab_path(str): 词汇表路径

# SimnetBow 服务部署

PaddleHub Serving可以部署一个在线语义匹配服务，可以将此接口用于在线web应用。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m simnet_bow  
```

启动时会显示加载模型过程，启动成功后显示
```shell
Loading simnet_bow successful.
```

这样就完成了服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import request
import json

# 待预测数据
test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

text = [test_text_1, test_text_2]

# 设置运行配置
# 对应本地预测simnet_bow.similarity(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":True}

# 指定预测方法为simnet_bow并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://HOST_IP:8866/predict/simnet_bow"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)
