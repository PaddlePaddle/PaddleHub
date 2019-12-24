# 部署情感分析服务-以senta_lstm为例
## 1 简介
&emsp;&emsp;情感分析针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度。利用`senta_lstm`模型可以完成中文情感分析任务，关于`senta_lstm`的具体信息请参阅[senta_lstm](https://paddlepaddle.org.cn/hubdetail?name=senta_lstm&en_category=SentimentAnalysis)。

&emsp;&emsp;使用PaddleHub-Serving可以部署一个在线情感分析服务，可以将此接口用于分析评论、智能客服等应用。

&emsp;&emsp;这里就带领大家使用PaddleHub-Serving，通过简单几步部署一个情感分析在线服务。

## 2 启动PaddleHub-Serving
&emsp;&emsp;启动命令如下
```shell
$ hub serving start -m senta_lstm  
```
&emsp;&emsp;启动时会显示加载模型过程，启动成功后显示
```shell
Loading senta_lstm successful.
```
&emsp;&emsp;这样就完成了一个词法分析服务化API的部署，默认端口号为8866。

## 3 测试词法分析在线API
&emsp;&emsp;在服务部署好之后，我们可以进行测试，用来测试的文本为`我不爱吃甜食`和`我喜欢躺在床上看电影`。

&emsp;&emsp;准备的数据格式为
```python
{"text": [text_1, text_2, ...]}  
```
&emsp;&emsp;注意字典的key为"text"。

&emsp;&emsp;根据文本和数据格式，代码如下
```python
>>> # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = ["我不爱吃甜食", "我喜欢躺在床上看电影"]
>>> text = {"text": text_list}
```
&emsp;&emsp;接下来发送请求到词法分析API，并得到结果，代码如下
```python
# 指定预测方法为lac并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/senta_lstm"
>>> r = requests.post(url=url, data=text)
```
&emsp;&emsp;`LAC`模型返回的结果为每个文本分词后的结果，我们尝试打印接口返回结果
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
&emsp;&emsp;这样我们就完成了对词法分析的预测服务化部署和测试。

&emsp;&emsp;完整的测试代码见[lac_serving_demo.py](./lac_serving_demo.py)。

### 3.2 使用自定义词典
`LAC`模型在预测时还可以使用自定义词典干预默认分词结果，这种情况只需要将自定义词典以文件的形式附加到request请求即可，数据格式如下
```python
{"user_dict": user_dict.txt}
```
根据数据格式，具体代码如下
```python
>>> # 指定自定义词典{"user_dict": dict.txt}
>>> file = {"user_dict": open("dict.txt", "rb")}
>>> # 请求接口时以文件的形式附加自定义词典，其余和不使用自定义词典的请求方式相同，此处不再赘述
>>> url = "http://127.0.0.1:8866/predict/text/lac"
>>> r = requests.post(url=url, files=file, data=text)
```
&emsp;&emsp;完整的测试代码见[lac_with_dict_serving_demo.py](./lac_with_dict_serving_demo.py)。




















## 数据格式  
input: {"text": [text_1, text_2, ...]}  
output: {"results":[result_1, result_2, ...]}  

## Serving快速启动命令  
```shell
$ hub serving start -m senta_lstm  
```

## python脚本
``` shell
$ python senta_lstm_serving_demo.py  
```  

## 结果示例  
```python
{  
    "results": [  
        {  
            "negative_probs": 0.7079,  
            "positive_probs": 0.2921,  
            "sentiment_key": "negative",  
            "sentiment_label": 0,  
            "text": "我不爱吃甜食"  
        },  
        {  
            "negative_probs": 0.0149,  
            "positive_probs": 0.9851,  
            "sentiment_key": "positive",  
            "sentiment_label": 1,  
            "text": "我喜欢躺在床上看电影"  
        }  
    ]  
}  
```
