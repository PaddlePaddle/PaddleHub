# 部署文本审核服务-以porn_detection_lstm为例
## 简介
在网站建设等场景中经常需要对敏感信息进行鉴定和过滤，采用文本审核模型`porn_detection_lstm`可自动判别文本是否涉黄并给出相应的置信度，关于`porn_detection_lstm`的具体信息请参见[porn_detection_lstm](https://paddlepaddle.org
.cn/hubdetail?name=porn_detection_lstm&en_category=TextCensorship)

使用PaddleHub Serving可以部署一个在线文本审核服务，可以将此接口用于防止低俗交友、色情文本等应用。

这里就带领大家使用PaddleHub Serving，通过简单几步部署一个文本审核在线服务。

## Step1：启动PaddleHub Serving
启动命令如下：
```shell
$ hub serving start -m porn_detection_lstm  
```
启动时会显示加载模型过程，启动成功后显示：
```shell
Loading porn_detection_lstm successful.
```
这样就完成了一个文本审核服务化API的部署，默认端口号为8866。

## Step2：测试文本审核在线API
在服务部署好之后，我们可以进行测试，用来测试的文本为`黄片下载`和`中国黄页`。

准备的数据格式为：
```python
{"text": [text_1, text_2, ...]}  
```
**NOTE:** 字典的key为"text"。

根据文本和数据格式，代码如下：
```python
>>> # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = ["黄片下载", "中国黄页"]
>>> text = {"text": text_list}
```
## Step3：获取并验证结果
接下来发送请求到文本审核API，并得到结果，代码如下：
```python
# 指定预测方法为lac并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/porn_detection_lstm"
>>> r = requests.post(url=url, data=text)
```
`porn_detection_lstm`模型返回的结果为每个文本鉴定后的结果，我们尝试打印接口返回结果：
```python
# 打印预测结果
>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))
{
    "results": [
        {
            "not_porn_probs": 0.0121,
            "porn_detection_key": "porn",
            "porn_detection_label": 1,
            "porn_probs": 0.9879,
            "text": "黄片下载"
        },
        {
            "not_porn_probs": 0.9954,
            "porn_detection_key": "not_porn",
            "porn_detection_label": 0,
            "porn_probs": 0.0046,
            "text": "中国黄页"
        }
    ]
}
```
可以看出正确得到了两个文本的预测结果。

这样我们就完成了对文本审核模型的预测服务化部署和测试。

完整的测试代码见[porn_detection_lstm_serving_demo.py](porn_detection_lstm_serving_demo.py)。

## 客户端请求新版模型的方式
对某些新版模型，客户端请求方式有所变化，更接近本地预测的请求方式，以降低学习成本。
以lac(2.1.0)为例，使用上述方法进行请求将提示：
```python
{
    "Warnning": "This usage is out of date, please use 'application/json' as content-type to post to /predict/lac. See 'https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.6/docs/tutorial/serving.md' for more details."
}
```
对于lac(2.1.0)，请求的方式如下：
```python
# coding: utf8
import requests
import json

if __name__ == "__main__":
    # 指定用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
    text = ["今天是个好日子", "天气预报说今天要下雨"]
    # 以key的方式指定text传入预测方法的时的参数，此例中为"texts"
    # 对应本地部署，则为lac.analysis_lexical(texts=[text1, text2])
    data = {"texts": text}
    # 指定预测方法为lac并发送post请求
    url = "http://127.0.0.1:8866/predict/lac"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

此Demo的具体信息和代码请参见[LAC Serving_2.1.0](../../demo/serving/module_serving/lexical_analysis_lac/lac_2.1.0_serving_demo.py)。
