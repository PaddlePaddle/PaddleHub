# 如何编写一个PaddleHub Module

## 模型基本信息

我们准备编写一个用于做情感分析的PaddleHub Module，Module的基本信息如下：
```yaml
name: senta_test
version: 1.0.0
summary: This is a PaddleHub Module. Just for test.
author: anonymous
author_email:
type: nlp/sentiment_analysis
```

Module存在一个接口sentiment_classify，用于接收传入文本，并给出文本的情感倾向（正面/负面），支持python接口调用和命令行调用。
```python
import paddlehub as hub

senta_test = hub.Module(name="senta_test")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```
```cmd
hub run senta_test --input_text 这部电影太差劲了
```

<br/>

## 策略

为了示例代码简单起见，我们使用一个非常简单的情感判断策略，当输入文本中带有词表中指定单词时，则判断文本倾向为负向，否则为正向

<br/>

## Module创建

### step 1. 创建必要的目录与文件

创建一个senta_test的目录，并在senta_test目录下分别创建__init__.py、module.py、processor.py、vocab.list，其中

|文件名|用途|
|-|-|
|\_\_init\_\_.py|空文件|
|module.py|主模块，提供Module的实现代码|
|processor.py|辅助模块，提供词表加载的方法|
|vocab.list|存放词表|

```cmd
➜  tree senta_test
senta_test/
├── vocab.list
├── __init__.py
├── module.py
└── processor.py
```
### step 2. 实现辅助模块processor

在processor.py中实现一个load_vocab接口用于读取词表
```python
def load_vocab(vocab_path):
    with open(vocab_path) as file:
        return file.read().split()
```

### step 3. 编写Module处理代码

module.py文件为Module的入口代码所在，我们需要在其中实现预测逻辑。

#### step 3_1. 引入必要的头文件
```python
import argparse
import os

import paddlehub as hub
from paddlehub.module.module import runnable, moduleinfo

from senta_test.processor import load_vocab
```
`注意`：当引用Module中模块时，需要输入全路径，如senta_test.processor
#### step 3_2. 定义SentaTest类
module.py中需要有一个继承了hub.Module的类存在，该类负责实现预测逻辑，并使用moduleinfo填写基本信息。当使用hub.Module(name="senta_test")加载Module时，PaddleHub会自动创建SentaTest的对象并返回。
```python
@moduleinfo(
    name="senta_test",
    version="1.0.0",
    summary="This is a PaddleHub Module. Just for test.",
    author="anonymous",
    author_email="",
    type="nlp/sentiment_analysis",
)
class SentaTest(hub.Module):
    ...
```
#### step 3_3. 执行必要的初始化
```python
def _initialize(self):
    # add arg parser
    self.parser = argparse.ArgumentParser(
        description="Run the senta_test module.",
        prog='hub run senta_test',
        usage='%(prog)s',
        add_help=True)
    self.parser.add_argument(
        '--input_text', type=str, default=None, help="text to predict")

    # load word dict
    vocab_path = os.path.join(self.directory, "vocab.list")
    self.vocab = load_vocab(vocab_path)
```
`注意`：执行类的初始化不能使用默认的__init__接口，而是应该重载实现_initialize接口。对象默认内置了directory属性，可以直接获取到Module所在路径
#### step 3_4. 完善预测逻辑
```python
def sentiment_classify(self, texts):
    results = []
    for text in texts:
        sentiment = "positive"
        for word in self.vocab:
            if word in text:
                sentiment = "negative"
                break
        results.append({"text":text, "sentiment":sentiment})

    return results
```
#### step 3_5. 支持命令行调用
如果希望Module可以支持命令行调用，则需要提供一个经过runnable修饰的接口，接口负责解析传入数据并进行预测，将结果返回。

如果不需要提供命令行预测功能，则可以不实现该接口，PaddleHub在用命令行执行时，会自动发现该Module不支持命令行方式，并给出提示。
```python
@runnable
def run_cmd(self, argvs):
    args = self.parser.parse_args(argvs)
    texts = [args.input_text]
    return self.sentiment_classify(texts)
```
#### step 3_6. 支持serving调用

TODO

### 完整代码

* [module.py](./senta_test/module.py)

* [processor.py](./senta_test/module.py)

<br/>

## 测试步骤

完成Module编写后，我们可以通过以下方式测试该Module

### 调用方法1

将Module安装到本机中，再通过Hub.Module(name=...)加载
```shell
hub install senta_test
```

```python
import paddlehub as hub

senta_test = hub.Module(name="senta_test")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```

### 调用方法2

直接通过Hub.Module(directory=...)加载
```python
import paddlehub as hub

senta_test = hub.Module(directory="senta_test/")
senta_test.sentiment_classify(texts=["这部电影太差劲了"])
```

### 调用方法3
将senta_test作为路径加到环境变量中，直接加载SentaTest对象
```shell
export PYTHONPATH=senta_test:$PYTHONPATH
```

```python
from senta_test.module import SentaTest

SentaTest.sentiment_classify(texts=["这部电影太差劲了"])
```

### 调用方法4
将Module安装到本机中，再通过hub run运行
```shell
hub install senta_test
hub run senta_test --input_text "这部电影太差劲了"
```
