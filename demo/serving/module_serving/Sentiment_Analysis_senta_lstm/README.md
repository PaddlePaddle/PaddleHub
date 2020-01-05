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

准备的数据格式为
```python
{"text": [text_1, text_2, ...]}  
```
注意字典的key为"text"。

根据文本和数据格式，代码如下
```python
>>> # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
>>> text = {"text": text_list}
```

## Step3：获取并验证结果
接下来发送请求到词法分析API，并得到结果，代码如下
```python
# 指定预测方法为lac并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/senta_lstm"
>>> r = requests.post(url=url, data=text)
```
`LAC`模型返回的结果为每个文本分词后的结果，我们尝试打印接口返回结果
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
