# albert-base-v1
|模型名称|albert-base-v1|
| :--- | :---: |
|类别|文本-语义模型|
|网络|albert-base-v1|
|数据集|-|
|是否支持Fine-tuning|是|
|模型大小|90MB|
|最新更新日期|2022-02-08|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - ALBERT针对当前预训练模型参数量过大的问题，提出了以下改进方案：

    - 嵌入向量参数化的因式分解。ALBERT对词嵌入参数进行了因式分解，先将单词映射到一个低维的词嵌入空间E，然后再将其映射到高维的隐藏空间H。

    - 跨层参数共享。ALBERT共享了层之间的全部参数。

更多详情请参考[ALBERT论文](https://arxiv.org/abs/1909.11942)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install albert-base-v1
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测  

- ### 1、预测代码示例

```python
import paddlehub as hub

data = [
    ['这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般'],
    ['怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片'],
    ['作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。'],
]
label_map = {0: 'negative', 1: 'positive'}

model = hub.Module(
    name='albert-base-v1',
    version='1.0.0',
    task='seq-cls',
    load_checkpoint='/path/to/parameters',
    label_map=label_map)
results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)
for idx, text in enumerate(data):
    print('Data: {} \t Label: {}'.format(text, results[idx]))
```

详情可参考PaddleHub示例：
- [文本分类](../../../../demo/text_classification)
- [序列标注](../../../../demo/sequence_labeling)

- ### 2、API

  - ```python
    def __init__(
        task=None,
        load_checkpoint=None,
        label_map=None,
        num_classes=2,
        suffix=False,
        **kwargs,
    )
    ```

    - 创建Module对象（动态图组网版本）

    - **参数**

      - `task`： 任务名称，可为`seq-cls`(文本分类任务)或`token-cls`(序列标注任务)。
      - `load_checkpoint`：使用PaddleHub Fine-tune api训练保存的模型参数文件路径。
      - `label_map`：预测时的类别映射表。
      - `num_classes`：分类任务的类别数，如果指定了`label_map`，此参数可不传，默认2分类。
      - `suffix`: 序列标注任务的标签格式，如果设定为`True`，标签以'-B', '-I', '-E' 或者 '-S'为结尾，此参数默认为`False`。
      - `**kwargs`：用户额外指定的关键字字典类型的参数。

  - ```python
    def predict(
        data,
        max_seq_len=128,
        batch_size=1,
        use_gpu=False
    )
    ```

    - **参数**

      - `data`： 待预测数据，格式为\[\[sample\_a\_text\_a, sample\_a\_text\_b\], \[sample\_b\_text\_a, sample\_b\_text\_b\],…,\]，其中每个元素都是一个样例，每个样例可以包含text\_a与text\_b。每个样例文本数量（1个或者2个）需和训练时保持一致。
      - `max_seq_len`：模型处理文本的最大长度
      - `batch_size`：模型批处理大小
      - `use_gpu`：是否使用gpu，默认为False。对于GPU用户，建议开启use_gpu。

    - **返回**

      - `results`：list类型，不同任务类型的返回结果如下
        - 文本分类：列表里包含每个句子的预测标签，格式为\[label\_1, label\_2, …,\]
        - 序列标注：列表里包含每个句子每个token的预测标签，格式为\[\[token\_1, token\_2, …,\], \[token\_1, token\_2, …,\], …,\]

  - ```python
    def get_embedding(
      data,
      use_gpu=False
    )
    ```

    - 用于获取输入文本的句子粒度特征与字粒度特征

    - **参数**

      - `data`：输入文本列表，格式为\[\[sample\_a\_text\_a, sample\_a\_text\_b\], \[sample\_b\_text\_a, sample\_b\_text\_b\],…,\]，其中每个元素都是一个样例，每个样例可以包含text\_a与text\_b。
      - `use_gpu`：是否使用gpu，默认为False。对于GPU用户，建议开启use_gpu。

    - **返回**

      - `results`：list类型，格式为\[\[sample\_a\_pooled\_feature, sample\_a\_seq\_feature\], \[sample\_b\_pooled\_feature, sample\_b\_seq\_feature\],…,\]，其中每个元素都是对应样例的特征输出，每个样例都有句子粒度特征pooled\_feature与字粒度特征seq\_feature。  

## 四、服务部署

- PaddleHub Serving可以部署一个在线获取预训练词向量。

- ### 第一步：启动PaddleHub Serving

  - ```shell
    $ hub serving start -m albert-base-v1
    ```

  - 这样就完成了一个获取预训练词向量服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 指定用于获取embedding的文本[[text_1], [text_2], ... ]}
    text = [["今天是个好日子"], ["天气预报说今天要下雨"]]
    # 以key的方式指定text传入预测方法的时的参数，此例中为"data"
    # 对应本地部署，则为module.get_embedding(data=text)
    data = {"data": text}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/albert-base-v1"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```

## 五、更新历史

* 1.0.0

  初始发布
