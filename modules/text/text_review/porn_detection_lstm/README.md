# PornDetectionLSTM API说明

## detection(texts=[], data={}, use_gpu=False, batch_size=1)

porn_detection_lstm预测接口，鉴定输入句子是否为包含色情文案

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小

**返回**

* results(list): 鉴定结果

## context(trainable=False)

获取porn_detection_lstm的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

## get_labels()

获取porn_detection_lstm的类别

**返回**

* labels(dict): porn_detection_lstm的类别

## get_vocab_path()

获取预训练时使用的词汇表

**返回**

* vocab_path(str): 词汇表路径

# PornDetectionLSTM 服务部署

PaddleHub Serving可以部署一个在线色情文案检测服务，可以将此接口用于在线web应用。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m porn_detection_lstm
```

启动时会显示加载模型过程，启动成功后显示
```shell
Loading porn_detection_lstm successful.
```

这样就完成了服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import request
import json

# 待预测数据
text = ["黄片下载", "打击黄牛党"]

# 设置运行配置
# 对应本地预测porn_detection_lstm.detection(texts=text, batch_size=1, use_gpu=True)
data = {"texts": text, "batch_size": 1, "use_gpu":True}

# 指定预测方法为porn_detection_lstm并发送post请求，content-type类型应指定json方式
# HOST_IP为服务器IP
url = "http://HOST_IP:8866/predict/porn_detection_lstm"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关于PaddleHub Serving更多信息参考[服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md)
