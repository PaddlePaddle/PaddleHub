# 部署语义模型服务-以simnet_bow为例
## 简介
`simnet_bow`是一个计算短文本相似度的模型，可以根据用户输入的两个文本，计算出相似度得分。关于`simnet_bow`的具体信息请参见[simnet_bow](https://paddlepaddle.org.cn/hubdetail?name=simnet_bow&en_category=SemanticModel)。

使用PaddleHub Serving可以部署一个在线语义模型服务，可以将此接口用于在线文本相似度分析、智能问答检索等应用。

这里就带领大家使用PaddleHub Serving，通过简单几步部署一个语义模型在线服务。

## Step1：启动PaddleHub Serving
启动命令如下
```shell
$ hub serving start -m simnet_bow  
```
启动时会显示加载模型过程，启动成功后显示
```shell
Loading lac successful.
```
这样就完成了一个语义模型服务化API的部署，默认端口号为8866。

## Step2：测试语义模型在线API
在服务部署好之后，我们可以进行测试，用来测试的文本对分别为`[这道题太难了:这道题是上一年的考题], [这道题太难了:这道题不简单], [这道题太难了:这道题很有意思]`。

准备的数据格式为
```python
{"text_1": [text_a1, text_a2, ... ], "text_2": [text_b1, text_b2, ... ]}
```
注意字典的key分别为"text_1"和"text_2"，与`simnet_bow`模型使用的输入数据一致。

根据文本和数据格式，代码如下
```python
>>> # 指定用于用于匹配的文本并生成字典{"text_1": [text_a1, text_a2, ... ]
>>> #                              "text_2": [text_b1, text_b2, ... ]}
>>> text = {
>>>     "text_1": ["这道题太难了", "这道题太难了", "这道题太难了"],
>>>     "text_2": ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
>>> }
```

## Step3：获取并验证结果
接下来发送请求到语义模型API，并得到结果，代码如下
```python
>>> # 指定匹配方法为simnet_bow并发送post请求
>>> url = "http://127.0.0.1:8866/predict/text/simnet_bow"
>>> r = requests.post(url=url, data=text)
```
`simnet_bow`模型返回的结果为每对文本对比后的相似度，我们尝试打印接口返回结果
```python
# 打印预测结果
>>> print(json.dumps(r.json(), indent=4, ensure_ascii=False))
{
    "results": [
        {
            "similarity": 0.8445,
            "text_1": "这道题太难了",
            "text_2": "这道题是上一年的考题"
        },
        {
            "similarity": 0.9275,
            "text_1": "这道题太难了",
            "text_2": "这道题不简单"
        },
        {
            "similarity": 0.9083,
            "text_1": "这道题太难了",
            "text_2": "这道题很有意思"
        }
    ]
}
```
这样我们就完成了对语义模型simnet_bow的预测服务化部署和测试。

完整的测试代码见[simnet_bow_serving_demo.py](simnet_bow_serving_demo.py)。
