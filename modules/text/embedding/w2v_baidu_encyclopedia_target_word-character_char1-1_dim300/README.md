# w2v_baidu_encyclopedia_target_word-character_char1-1_dim300
|模型名称|w2v_baidu_encyclopedia_target_word-character_char1-1_dim300|
| :--- | :---: | 
|类别|文本-词嵌入|
|网络|w2v|
|数据集|baidu_encyclopedia|
|是否支持Fine-tuning|否|
|文件大小|679.15MB|
|词表大小|636038|
|最新更新日期|2021-04-28|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - PaddleHub提供多个开源的预训练Embedding模型。这些Embedding模型可根据不同语料、不同训练方式和不同的维度进行区分，关于模型的具体信息可参考PaddleNLP的文档：[Embedding模型汇总](https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/docs/embeddings.md)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install w2v_baidu_encyclopedia_target_word-character_char1-1_dim300
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    embedding = hub.Module(name='w2v_baidu_encyclopedia_target_word-character_char1-1_dim300')

    # 获取单词的embedding
    embedding.search("中国")
    # 计算两个词向量的余弦相似度
    embedding.cosine_sim("中国", "美国")
    # 计算两个词向量的内积
    embedding.dot("中国", "美国")
    ```

- ### 2、API

  - ```python
    def __init__(
        *args,
        **kwargs
    )
    ```

    - 创建一个Embedding Module对象，默认无需参数。

    - **参数**
      - `*args`： 用户额外指定的列表类型的参数。
      - `**kwargs`：用户额外指定的关键字字典类型的参数。

    - 关于额外参数的详情可参考[paddlenlp.embeddings](https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleNLP/paddlenlp/embeddings)


  - ```python
    def search(
        words: Union[List[str], str, int],
    )
    ```

    - 获取一个或多个词的embedding。输入可以是`str`、`List[str]`和`int`类型，分别代表获取一个词，多个词和指定词编号的embedding，词的编号和模型的词典相关，词典可通过模型实例的`vocab`属性获取。

    - **参数**
      - `words`： 需要获取的词向量的词、词列表或者词编号。


  - ```python
    def cosine_sim(
        word_a: str,
        word_b: str,
    )
    ```

    - 计算两个词embedding的余弦相似度。需要注意的是`word_a`和`word_b`都需要是词典里的单词，否则将会被认为是OOV(Out-Of-Vocabulary)，同时被替换为`unknown_token`。

    - **参数**
      - `word_a`： 需要计算余弦相似度的单词a。
      - `word_b`： 需要计算余弦相似度的单词b。


  - ```python
    def dot(
        word_a: str,
        word_b: str,
    )
    ```

    - 计算两个词embedding的内积。对于输入单词同样需要注意OOV问题。

    - **参数**
      - `word_a`： 需要计算内积的单词a。
      - `word_b`： 需要计算内积的单词b。


  - ```python
    def get_vocab_path()
    ```

    - 获取本地词表文件的路径信息。


  - ```python
    def get_tokenizer(*args, **kwargs)
    ```

    - 获取当前模型的tokenizer，返回一个JiebaTokenizer的实例，当前只支持中文embedding模型。

    - **参数**
      - `*args`： 额外传递的列表形式的参数。
      - `**kwargs`： 额外传递的字典形式的参数。

    - 关于额外参数的详情可参考[paddlenlp.data.tokenizer.JiebaTokenizer](https://github.com/PaddlePaddle/models/blob/release/2.0-beta/PaddleNLP/paddlenlp/data/tokenizer.py)

  - 更多api详情和用法可参考[paddlenlp.embeddings](https://github.com/PaddlePaddle/models/tree/release/2.0-beta/PaddleNLP/paddlenlp/embeddings)


## 四、部署服务

- 通过PaddleHub Serving，可以部署一个在线获取两个词向量的余弦相似度的服务。

- ### Step1: 启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m w2v_baidu_encyclopedia_target_word-character_char1-1_dim300
    ```

  - 这样就完成了一个获取词向量的余弦相似度服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步: 发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 指定用于计算余弦相似度的单词对[[word_a, word_b], [word_a, word_b], ... ]]
    word_pairs = [["中国", "美国"], ["今天", "明天"]]
    # 以key的方式指定word_pairs传入预测方法的时的参数，此例中为"data"，对于每一对单词，调用cosine_sim进行余弦相似度的计算
    data = {"data": word_pairs}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/w2v_baidu_encyclopedia_target_word-character_char1-1_dim300"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  优化模型
  - ```shell
    $ hub install w2v_baidu_encyclopedia_target_word-character_char1-1_dim300==1.0.1
    ```
