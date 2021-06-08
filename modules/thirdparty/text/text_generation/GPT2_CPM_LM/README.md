## 概述
CPM-LM 是一个基于 GPT-2 的预训练生成模型，参数规模达 26 亿，预训练中文数据规模 100 GB，使用了 64 块 V100 GPU，训练时间约为 3 周。能够在多种自然语言处理任务上进行零次学习或少次学习，并达到较好的效果。基于给定上文，模型可以续写出一致性高、可读性强的文本，达到现有中文生成模型的领先效果。

模型参数转换至官方开源项目，由于模型较大，推荐在GPU环境下运行，并且请确保运行环境的内存大于20G且显卡显存大于12G，否则可能无法正常运行

更多详情参考[清源CPM官网](https://cpm.baai.ac.cn)及其[Github项目主页](https://github.com/TsinghuaAI/CPM-Generate)

## API
```python
def greedy_search(
    text,
    max_len=32,
    end_word=None):
```
文本生成 API ，根据输入的文字进行文本生成，使用 Greedy Search 进行解码，生成的文本单一且确定，适合于问答类的文本生成。

**参数**
* text (str) : 输入文本
* max_len (int) : 生成文本的最大长度
* end_word (str or None) : 终止生成的标志词

**返回**
* results (str): 生成的文本

```python
def nucleus_sample(
    text,
    max_len=32,
    end_word=None,
    repitition_penalty=1.0,
    temperature=1.0,
    top_k=0,
    top_p=1.0):
```
文本生成 API ，根据输入的文字进行文本生成，使用采样的方式进行解码，生成的文本比较多样，适合于文章类的文本生成。

**参数**
* text (str) : 输入文本
* max_len (int) : 生成文本的最大长度
* end_word (str or None) : 终止生成的标志词
* repitition_penalty (float) : 重复词抑制率，大于1抑制，小于1提高
* temperature (float) ：较低的temperature可以让模型对最佳选择越来越有信息，大于1，则会降低，0则相当于 argmax/max ，inf则相当于均匀采样
* top_k (int) : 抑制小于 Top K 的输出，大于0时有效
* top_p (float) : 抑制小于 Top P 的输出，小于1.0时有效

**返回**
* results (str): 生成的文本

**代码示例**
* 加载模型：
```python
import paddlehub as hub

model = hub.Module(name='GPT2_CPM_LM')
```
* 使用 Greedy Search 生成文本：
```python
inputs = '''默写古诗:
日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，'''

outputs = model.greedy_search(inputs, max_len=10, end_word='\n')

print(outputs)
```
    默写古诗:
    日照香炉生紫烟,遥看瀑布挂前川。
    飞流直下三千尺,疑是银河落九天。
```python
inputs = '''问题：西游记是谁写的？
答案：'''

outputs = model.greedy_search(inputs, max_len=10, end_word='\n')

print(outputs)
```
    问题:西游记是谁写的?
    答案:吴承恩。
```python
inputs = '''小明决定去吃饭，小红继续写作业
问题：去吃饭的人是谁？
答案：'''

outputs = model.greedy_search(inputs, max_len=10, end_word='\n')

print(outputs)
```
    小明决定去吃饭,小红继续写作业
    问题:去吃饭的人是谁?
    答案:小明
```python
inputs = '''默写英文：
狗：dog
猫：'''

outputs = model.greedy_search(inputs, max_len=10, end_word='\n')

print(outputs)
```
    默写英文:
    狗:dog
    猫:cat

* 使用采样方式生成文本：

```python
inputs = '''在此处输入文本的开头'''

outputs = model.nucleus_sample(
    inputs,
    max_len=32,
    end_word='。',
    repitition_penalty=1.0,
    temperature=1.0,
    top_k=5,
    top_p=1.0
)

print(outputs)
```
    在此处输入文本的开头、结尾或中间部分,然后按下回车键。


```python
inputs = '''《乡土中国》是费孝通先生在社区研究的基础上从宏观角度探讨中国社会结构的著作，'''

outputs = model.nucleus_sample(
    inputs,
    max_len=32,
    end_word='。',
    repitition_penalty=1.0,
    temperature=1.0,
    top_k=3000,
    top_p=1.0
)

print(outputs)
```
    《乡土中国》是费孝通先生在社区研究的基础上从宏观角度探讨中国社会结构的著作,在书中,他试图向读者说明,中国社会的结构是如何形成、演变的,家庭伦理是如何形成、



## 服务部署

PaddleHub Serving 可以部署一个在线文本生成服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start --modules GPT2_CPM_LM
```

这样就完成了一个文本生成的在线服务API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json

text = "今天是个好日子"
data = {
    "text": text,
    "mode": "sample", # 'search' or 'sample'
    # 可以更加需要设置上述 API 中提到的其他参数
}
url = "http://127.0.0.1:8866/predict/GPT2_CPM_LM"
headers = {"Content-Type": "application/json"}

r = requests.post(url=url, headers=headers, data=json.dumps(data))
```


## 查看代码
https://github.com/jm12138/CPM-Generate-Paddle

## 依赖
paddlepaddle >= 2.0.0

paddlehub >= 2.0.0

sentencepiece==0.1.92
