# 部署词法分析服务-以lac为例
## 简介
`Lexical Analysis of Chinese`，简称`LAC`，是一个联合的词法分析模型，能整体性地完成中文分词、词性标注、专名识别任务。关于`LAC`的具体信息请参见[LAC](https://paddlepaddle.org.cn/hubdetail?name=lac&en_category=LexicalAnalysis)。

使用PaddleHub Serving可以部署一个在线词法分析服务，可以将此接口用于词法分析、在线分词等在线web应用。

这里就带领大家使用PaddleHub Serving，通过简单几步部署一个词法分析在线服务。

## 2 启动PaddleHub Serving
启动命令如下
```shell
$ hub serving start -m lac  
```
启动时会显示加载模型过程，启动成功后显示
```shell
Loading lac successful.
```
这样就完成了一个词法分析服务化API的部署，默认端口号为8866。

## Step2：测试语言模型在线API
### 不使用自定义词典
在服务部署好之后，我们可以进行测试，用来测试的文本为`今天是个好日子`和`天气预报说今天要下雨`。

准备的数据格式为
```python
{"text": [text_1, text_2, ...]}  
```
注意字典的key为"text"。

根据文本和数据格式，代码如下
```python
>>> # 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = ["今天是个好日子", "天气预报说今天要下雨"]
>>> text = {"text": text_list}
```

## Step3：获取并验证结果
接下来发送请求到词法分析API，并得到结果，代码如下
```python
# 指定预测方法为lac并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/lac"
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

完整的测试代码见[lac_serving_demo.py](lac_serving_demo.py)。

### 使用自定义词典
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

完整的测试代码见[lac_with_dict_serving_demo.py](lac_with_dict_serving_demo.py)。
