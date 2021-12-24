# auto_punc

|模型名称|auto_punc|
| :--- | :---: |
|类别|文本-标点恢复|
|网络|Ernie-1.0|
|数据集|WuDaoCorpora 2.0|
|是否支持Fine-tuning|否|
|模型大小|568MB|
|最新更新日期|2021-12-24|
|数据指标|-|

## 一、模型基本信息

### 模型介绍

Ernie是百度提出的基于知识增强的持续学习语义理解模型，该模型将大数据预训练与多源丰富知识相结合，通过持续学习技术，不断吸收海量文本数据中词汇、结构、语义等方面的知识，实现模型效果不断进化。

["悟道"文本数据集](https://ks3-cn-beijing.ksyun.com/resources/WuDaoCorpora/WuDaoCorpora__A_Super_Large_scale_Chinese_Corporafor_Pre_training_Language_Models.pdf)
采用20多种规则从100TB原始网页数据中清洗得出最终数据集，注重隐私数据信息的去除，源头上避免GPT-3存在的隐私泄露风险；包含教育、科技等50+个行业数据标签，可以支持多领域预训练模型的训练。
- 数据总量：3TB
- 数据格式：json
- 开源数量：200GB
- 数据集下载：https://resource.wudaoai.cn/
- 日期：2021年12月23日

auto_punc采用了Ernie1.0预训练模型，在[WuDaoCorpora 2.0](https://resource.wudaoai.cn/home)的200G开源文本数据集上进行了标点恢复任务的训练，模型可直接用于预测，对输入的对中文文本自动添加7种标点符号：逗号（，）、句号（。）、感叹号（！）、问号（？）、顿号（、）、冒号（：）和分号（；）。

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ernie_network_1.png" hspace='10'/> <br />
</p>

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ernie_network_2.png" hspace='10'/> <br />
</p>


更多详情请参考
- [WuDaoCorpora: A Super Large-scale Chinese Corpora for Pre-training Language Models](https://ks3-cn-beijing.ksyun.com/resources/WuDaoCorpora/WuDaoCorpora__A_Super_Large_scale_Chinese_Corporafor_Pre_training_Language_Models.pdf)
- [ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/abs/1904.09223)


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.1.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install auto_punc
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(
        name='auto_punc',
        version='1.0.0')

    texts = [
        '今天的天气真好啊你下午有空吗我想约你一起去逛街',
        '我最喜欢的诗句是先天下之忧而忧后天下之乐而乐',
    ]
    punc_texts = model.add_puncs(texts)
    print(punc_texts)
    # ['我最喜欢的诗句是：先天下之忧而忧，后天下之乐而乐。', '今天的天气真好啊！你下午有空吗？我想约你一起去逛街。']
    ```

- ### 2、API
  - ```python
    def add_puncs(
        texts: Union[str, List[str]],
        max_length=256,
        device='cpu'
    )
    ```
    - 对输入的中文文本自动添加标点符号。

    - **参数**

      - `texts`：输入的中文文本，可为str或List[str]类型，预测时，中英文和数字以外的字符将会被删除。
      - `max_length`：模型预测时输入的最大长度，超过时文本会被截断，默认为256。
      - `device`：预测时使用的设备，默认为`cpu`，如需使用gpu预测，请设置为`gpu`。

    - **返回**

      - `punc_texts`：List[str]类型，返回添加标点后的文本列表。


## 四、服务部署

- PaddleHub Serving可以部署一个在线的文本标点添加的服务。

- ### 第一步：启动PaddleHub Serving

  - ```shell
    $ hub serving start -m auto_punc
    ```

  - 这样就完成了一个文本标点添加服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 输入的中文文本，中英文和数字之外的字符在模型预测前会被删除
    texts = [
        '今天的天气真好啊你下午有空吗我想约你一起去逛街',
        '我最喜欢的诗句是先天下之忧而忧后天下之乐而乐',
    ]

    # 以key的方式指定text传入预测方法的时的参数，此例中为"texts"
    data = {"texts": texts}

    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/auto_punc"

    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```

## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install auto_punc
  ```
