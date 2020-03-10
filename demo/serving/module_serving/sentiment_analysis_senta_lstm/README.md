# 部署情感分析服务-以senta_lstm为例
## 简介
情感分析针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。利用`senta_lstm`模型可以完成中文情感分析任务，关于`senta_lstm`的具体信息请参见[senta_lstm]
(https://paddlepaddle.org.cn/hubdetail?name=senta_lstm&en_category=SentimentAnalysis)。

使用PaddleHub Serving可以部署一个在线情感分析服务，可以将此接口用于分析评论、智能客服等应用。

这里就带领大家使用PaddleHub Serving，通过简单几步部署一个情感分析在线服务。

## Step1：启动PaddleHub Serving
启动命令如下
```shell
$ hub serving start -m senta_lstm  
```
启动时会显示加载模型过程，启动成功后显示
```shell
Loading senta_lstm successful.
```
这样就完成了一个词法分析服务化API的部署，默认端口号为8866。

## Step2：测试词法分析在线API
在服务部署好之后，我们可以进行测试，用来测试的文本为`我不爱吃甜食`和`我喜欢躺在床上看电影`。

准备的数据格式为：
```python
{"text": [text_1, text_2, ...]}  
```
**NOTE:** 字典的key为"text"。

根据文本和数据格式，代码如下：
```python
>>> # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
>>> text = {"text": text_list}
```

## Step3：获取并验证结果
接下来发送请求到词法分析API，并得到结果，代码如下：
```python
# 指定预测方法为lac并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/senta_lstm"
>>> r = requests.post(url=url, data=text)
```
`LAC`模型返回的结果为每个文本分词后的结果，我们尝试打印接口返回结果：
```python
# 打印预测结果
>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))
{
    "results": [
        {
            "tag": [
                "TIME",
                "v",
                "q",
                "n"
            ],
            "word": [
                "今天",
                "是",
                "个",
                "好日子"
            ]
        },
        {
            "tag": [
                "n",
                "v",
                "TIME",
                "v",
                "v"
            ],
            "word": [
                "天气预报",
                "说",
                "今天",
                "要",
                "下雨"
            ]
        }
    ]
}
```
这样我们就完成了对词法分析的预测服务化部署和测试。

完整的测试代码见[senta_lstm_serving_demo.py](senta_lstm_serving_demo.py)。

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
