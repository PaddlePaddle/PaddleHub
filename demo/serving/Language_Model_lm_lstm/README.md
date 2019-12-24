# 部署语言模型服务-以lm_lstm为例
## 1 简介
&emsp;&emsp;利用语言模型`lm_lstm`能够评测一句话的流利程度。关于`lm_lstm`的具体信息请参阅[lm_lstm](https://paddlepaddle.org.cn/hubdetail?name=lm_lstm&en_category=LanguageModel)。

&emsp;&emsp;使用PaddleHub-Serving可以部署一个在线语言模型服务，可以将此接口用于文本分析等在线web应用。

&emsp;&emsp;这里就带领大家使用PaddleHub-Serving，通过简单几步部署一个语言模型在线服务。

## 2 启动PaddleHub-Serving
&emsp;&emsp;启动命令如下
```shell
$ hub serving start -m lm_lstm  
```
&emsp;&emsp;启动时会显示加载模型过程，启动成功后显示
```shell
Loading lm_lstm successful.
```
&emsp;&emsp;这样就完成了一个语言模型服务化API的部署，默认端口号为8866。

## 3 测试语言模型在线API
&emsp;&emsp;我们用来测试的文本为`the plant which is owned by <unk> & <unk> co. was under contract with <unk> to make the cigarette filter`，以及`more common <unk> fibers are <unk> and are more easily rejected by the body dr. <unk> explained`  

&emsp;&emsp;准备的数据格式为
```python
{"text": [text_1, text_2, ...]}  
```
&emsp;&emsp;注意字典的key为"text"。

&emsp;&emsp;根据文本和数据格式，代码如下
```python
# 指定用于用于预测的文本并生成字典{"text": [text_1, text_2, ... ]}
>>> text_list = [
...     "the plant which is owned by <unk> & <unk> co. was under contract with <unk> to make the cigarette filter",
...     "more common <unk> fibers are <unk> and are more easily rejected by the body dr. <unk> explained"
... ]
>>> text = {"text": text_list}
```
&emsp;&emsp;接下来发送请求到语言模型API，并得到结果，代码如下
```python
# 指定预测方法为lm_lstm并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/lm_lstm"
>>> r = requests.post(url=url, data=text)
```
&emsp;&emsp;lm_lstm模型返回的结果为每个文本及其对应的困扰度，我们尝试打印接口返回结果
```python
# 打印预测结果
>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))
{
    "results": [
        {
            "perplexity": 4.584166451916099,
            "text": "the plant which is owned by <unk> & <unk> co. was under contract with <unk> to make the cigarette filter"
        },
        {
            "perplexity": 6.038358983397484,
            "text": "more common <unk> fibers are <unk> and are more easily rejected by the body dr. <unk> explained"
        }
    ]
}
```
&emsp;&emsp;这样我们就完成了对语言模型的预测服务化部署和测试。

&emsp;&emsp;完整的测试代码见[lm_lstm_serving_demo.py](./lm_lstm_serving_demo.py)。
