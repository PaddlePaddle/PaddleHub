# transformer_en-de
|模型名称|transformer_en-de|
| :--- | :---: | 
|类别|文本-机器翻译|
|网络|Transformer|
|数据集|WMT14 EN-DE|
|是否支持Fine-tuning|否|
|模型大小|481MB|
|最新更新日期|2021-07-21|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - 2017 年，Google机器翻译团队在其发表的论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)中，提出了用于完成机器翻译（Machine Translation）等序列到序列（Seq2Seq）学习任务的一种全新网络结构——Transformer。Tranformer网络完全使用注意力（Attention）机制来实现序列到序列的建模，并且取得了很好的效果。

  - transformer_en-de包含6层的transformer结构，头数为8，隐藏层参数为512，参数量为64M。该模型在[WMT'14 EN-DE数据集](http://www.statmt.org/wmt14/translation-task.html)进行了预训练，加载后可直接用于预测，提供了英文翻译为德文的能力。

  - 关于机器翻译的Transformer模型训练方式和详情，可查看[Machine Translation using Transformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/machine_translation/transformer)。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0
  
  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install transformer_en-de
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub 

    model = hub.Module(name='transformer_en-de', beam_size=5)
    src_texts = [
        'What are you doing now?',
        'The change was for the better; I eat well, I exercise, I take my drugs.',
        'Such experiments are not conducted for ethical reasons.',
    ]

    n_best = 3  # 每个输入样本的输出候选句子数量
    trg_texts = model.predict(src_texts, n_best=n_best)
    for idx, st in enumerate(src_texts):
        print('-'*30)
        print(f'src: {st}')
        for i in range(n_best):
            print(f'trg[{i+1}]: {trg_texts[idx*n_best+i]}')
    ```

- ### 2、API

  - ```python
    def __init__(max_length: int = 256,
             max_out_len: int = 256,
             beam_size: int = 5):
    ```

    - 初始化module，可配置模型的输入输出文本的最大长度和解码时beam search的宽度。

    - **参数**

      - `max_length`(int): 输入文本的最大长度，默认值为256。
      - `max_out_len`(int): 输出文本的最大解码长度，默认值为256。
      - `beam_size`(int): beam search方式解码的beam宽度，默认为5。

  - ```python
    def predict(data: List[str],
            batch_size: int = 1,
            n_best: int = 1,
            use_gpu: bool = False):
    ```

    - 预测API，输入源语言的文本句子，解码后输出翻译后的目标语言的文本候选句子。

    - **参数**

      - `data`(List[str]): 源语言的文本列表，数据类型为List[str]
      - `batch_size`(int): 进行预测的batch_size，默认为1
      - `n_best`(int): 每个输入文本经过模型解码后，输出的得分最高的候选句子的数量，必须小于beam_size，默认为1
      - `use_gpu`(bool): 是否使用gpu执行预测，默认为False
    
    - **返回**

      - `results`(List[str]): 翻译后的目标语言的候选句子，长度为`len(data)*n_best`

## 四、服务部署

- 通过启动PaddleHub Serving，可以加载模型部署在线翻译服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m transformer_en-de
    ```

  - 通过以上命令可完成一个英德机器翻译API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    texts = [
        'What are you doing now?',
        'The change was for the better; I eat well, I exercise, I take my drugs.',
        'Such experiments are not conducted for ethical reasons.',
    ]
    data = {"data": texts}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/transformer_en-de"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  修复模型初始化的兼容性问题
  - ```shell
    $ hub install transformer_en-de==1.0.1
    ```
