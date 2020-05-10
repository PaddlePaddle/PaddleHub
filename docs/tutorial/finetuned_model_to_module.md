# Fine-tune保存的模型如何转化为一个PaddleHub Module

## 模型基本信息

本示例以模型ERNIE Tiny在数据集ChnSentiCorp上完成情感分类Fine-tune任务后保存的模型转化为一个PaddleHub Module，Module的基本信息如下：
```yaml
name: ernie_tiny_finetuned
version: 1.0.0
summary: ERNIE tiny which was fine-tuned on the chnsenticorp dataset.
author: anonymous
author_email:
type: nlp/semantic_model
```

**本示例代码可以参考[finetuned_model_to_module](../../demo/text_classification/finetuned_model_to_module/)**

Module存在一个接口predict，用于接收带预测，并给出文本的情感倾向（正面/负面），支持python接口调用和命令行调用。
```python
import paddlehub as hub

ernie_tiny_finetuned = hub.Module(name="ernie_tiny_finetuned")
ernie_tiny_finetuned.predcit(data=[["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
            ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]])
```


## Module创建

### step 1. 创建必要的目录与文件

创建一个finetuned_model_to_module的目录，并在finetuned_model_to_module目录下分别创建__init__.py、module.py，其中

|文件名|用途|
|-|-|
|\_\_init\_\_.py|空文件|
|module.py|主模块，提供Module的实现代码|
|ckpt文件|利用PaddleHub Fine-tune得到的ckpt文件夹，其中必须包含best_model文件|


```cmd
➜  tree finetuned_model_to_module
finetuned_model_to_module/
├── __init__.py
├── ckpt_chnsenticorp
│   ├── ***
│   ├── best_model
│   │   ├── ***
└── module.py
```

### step 2. 编写Module处理代码

module.py文件为Module的入口代码所在，我们需要在其中实现预测逻辑。

#### step 2_1. 引入必要的头文件
```python
import os

import numpy as np
from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
import paddlehub as hub
```

#### step 2_2. 定义ERNIE_Tiny_Finetuned类
module.py中需要有一个继承了hub.Module的类存在，该类负责实现预测逻辑，并使用moduleinfo填写基本信息。当使用hub.Module(name="ernie_tiny_finetuned")加载Module时，PaddleHub会自动创建ERNIE_Tiny_Finetuned的对象并返回。
```python
@moduleinfo(
    name="ernie_tiny_finetuned",
    version="1.0.0",
    summary="ERNIE tiny which was fine-tuned on the chnsenticorp dataset.",
    author="anonymous",
    author_email="",
    type="nlp/semantic_model")
class ERNIETinyFinetuned(hub.Module):
    ...
```
#### step 2_3. 执行必要的初始化
```python
def _initialize(self,
                ckpt_dir="ckpt_chnsenticorp",
                num_class=2,
                max_seq_len=128,
                use_gpu=False,
                batch_size=1):
    self.ckpt_dir = os.path.join(self.directory, ckpt_dir)
    self.num_class = num_class
    self.MAX_SEQ_LEN = max_seq_len

    self.params_path = os.path.join(self.ckpt_dir, 'best_model')
    if not os.path.exists(self.params_path):
        logger.error(
            "%s doesn't contain the best_model file which saves the best parameters as fietuning."
        )
        exit()

    # Load Paddlehub ERNIE Tiny pretrained model
    self.module = hub.Module(name="ernie_tiny")
    inputs, outputs, program = self.module.context(
        trainable=True, max_seq_len=max_seq_len)

    self.vocab_path = self.module.get_vocab_path()

    # Download dataset and use accuracy as metrics
    # Choose dataset: GLUE/XNLI/ChinesesGLUE/NLPCC-DBQA/LCQMC
    # metric should be acc, f1 or matthews
    metrics_choices = ["acc"]

    # For ernie_tiny, it use sub-word to tokenize chinese sentence
    # If not ernie tiny, sp_model_path and word_dict_path should be set None
    reader = hub.reader.ClassifyReader(
        vocab_path=self.module.get_vocab_path(),
        max_seq_len=max_seq_len,
        sp_model_path=self.module.get_spm_path(),
        word_dict_path=self.module.get_word_dict_path())

    # Construct transfer learning network
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_output" for token-level output.
    pooled_output = outputs["pooled_output"]

    # Setup feed list for data feeder
    # Must feed all the tensor of module need
    feed_list = [
        inputs["input_ids"].name,
        inputs["position_ids"].name,
        inputs["segment_ids"].name,
        inputs["input_mask"].name,
    ]

    # Setup runing config for PaddleHub Finetune API
    config = hub.RunConfig(
        use_data_parallel=False,
        use_cuda=use_gpu,
        batch_size=batch_size,
        checkpoint_dir=self.ckpt_dir,
        strategy=hub.AdamWeightDecayStrategy())

    # Define a classfication finetune task by PaddleHub's API
    self.cls_task = hub.TextClassifierTask(
        data_reader=reader,
        feature=pooled_output,
        feed_list=feed_list,
        num_classes=self.num_class,
        config=config,
        metrics_choices=metrics_choices)
```

初始化过程即为Fine-tune时创建Task的过程。

**NOTE:**
1. 执行类的初始化不能使用默认的__init__接口，而是应该重载实现_initialize接口。对象默认内置了directory属性，可以直接获取到Module所在路径。
2. 使用Fine-tune保存的模型预测时，无需加载数据集Dataset，即Reader中的dataset参数可为None。

#### step 3_4. 完善预测逻辑
```python
def predict(self, data, return_result=False, accelerate_mode=True):
    """
    Get prediction results
    """
    run_states = self.cls_task.predict(
        data=data,
        return_result=return_result,
        accelerate_mode=accelerate_mode)
    results = [run_state.run_results for run_state in run_states]
    prediction = []
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        batch_result = batch_result.tolist()
        prediction += batch_result
    return prediction
```

#### step 3_5. 支持serving调用

如果希望Module可以支持PaddleHub Serving部署预测服务，则需要将预测接口predcit加上serving修饰(`@serving`)，接口负责解析传入数据并进行预测，将结果返回。

如果不需要提供PaddleHub Serving部署预测服务，则可以不需要加上serving修饰。

```python
@serving
def predict(self, data, return_result=False, accelerate_mode=True):
    """
    Get prediction results
    """
    run_states = self.cls_task.predict(
        data=data,
        return_result=return_result,
        accelerate_mode=accelerate_mode)
    results = [run_state.run_results for run_state in run_states]
    prediction = []
    for batch_result in results:
        # get predict index
        batch_result = np.argmax(batch_result, axis=2)[0]
        batch_result = batch_result.tolist()
        prediction += batch_result
    return prediction
```

### 完整代码

* [module.py](../../demo/text_classification/finetuned_model_to_module/module.py)

* [__init__.py](../../demo/text_classification/finetuned_model_to_module/__init__.py)

**NOTE:** `__init__.py`是空文件

## 测试步骤

完成Module编写后，我们可以通过以下方式测试该Module

### 调用方法1

将Module安装到本机中，再通过Hub.Module(name=...)加载
```shell
hub install finetuned_model_to_module
```

安装成功会显示**Successfully installed ernie_tiny_finetuned**

```python
import paddlehub as hub
import numpy as np


ernie_tiny = hub.Module(name="ernie_tiny_finetuned")

# Data to be prdicted
data = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
        ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]

predictions = ernie_tiny.predict(data=data)
for index, text in enumerate(data):
    print("%s\tpredict=%s" % (data[index][0], predictions[index]))
```

### 调用方法2

直接通过Hub.Module(directory=...)加载
```python
import paddlehub as hub
import numpy as np

ernie_tiny_finetuned = hub.Module(directory="finetuned_model_to_module/")

# Data to be prdicted
data = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
        ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]

predictions = ernie_tiny.predict(data=data)
for index, text in enumerate(data):
    print("%s\tpredict=%s" % (data[index][0], predictions[index]))
```

### 调用方法3
将finetuned_model_to_module作为路径加到环境变量中，直接加载ERNIETinyFinetuned对象
```shell
export PYTHONPATH=finetuned_model_to_module:$PYTHONPATH
```

```python
from finetuned_model_to_module.module import ERNIETinyFinetuned
import numpy as np

# Data to be prdicted
data = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
        ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]

predictions = ERNIETinyFinetuned.predict(data=data)
for index, text in enumerate(data):
    print("%s\tpredict=%s" % (data[index][0], predictions[index]))
```


### PaddleHub Serving调用方法

**第一步:启动预测服务**

```shell
hub serving start -m ernie_tiny_finetuned
```

**第二步:发送请求，获取预测结果**

通过如下脚本既可以发送请求：
```python
# coding: utf8
import requests
import json


# 待预测文本
texts = [["这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般"], ["交通方便；环境很好；服务态度很好 房间较小"],
        ["19天硬盘就罢工了~~~算上运来的一周都没用上15天~~~可就是不能换了~~~唉~~~~你说这算什么事呀~~~"]]
# key为'data', 对应着预测接口predict的参数data
data = {'data': texts}

# 指定模型为ernie_tiny_finetuned并发送post请求，且请求的headers为application/json方式
url = "http://127.0.0.1:8866/predict/ernie_tiny_finetuned"
headers = {"Content-Type": "application/json"}
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(json.dumps(r.json(), indent=4, ensure_ascii=False))
```

关与PaddleHub Serving更多信息参见[Hub Serving教程](../../docs/tutorial/serving.md)以及[Demo](../../demo/serving)
