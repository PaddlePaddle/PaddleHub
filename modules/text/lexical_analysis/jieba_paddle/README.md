# jieba_paddle

|模型名称|jieba_paddle|
| :--- | :---: |
|类别|文本-词法分析|
|网络|BiGRU+CRF|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|28KB|
|最新更新日期|2021-02-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - 该Module是jieba使用PaddlePaddle深度学习框架搭建的切词网络（双向GRU）。同时也支持jieba的传统切词方法，如精确模式、全模式、搜索引擎模式等切词模式，使用方法和jieba保持一致。

  - 更多信息参考：[jieba](https://github.com/fxsjy/jieba)



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.8.0

  - paddlehub >= 1.8.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install jieba_paddle
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run jieba_paddle --input_text "今天天气真好"
    ```
    或者
  - ```shell
    $ hub run senta_gru --input_file test.txt
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    jieba = hub.Module(name="jieba_paddle")

    results = jieba.cut("今天是个好日子", cut_all=False, HMM=True)
    print(results)

    # ['今天', '是', '个', '好日子']
    ```


- ### 3、API

  - ```python
    def cut(sentence, use_paddle=True, cut_all=False, HMM=True)
    ```
    - jieba_paddle预测接口，预测输入句子的分词结果

    - **参数**

      - sentence(str): 单句预测数据。
      - use_paddle(bool): 是否使用paddle模式（双向GRU）切词，默认为True。
      - cut_all 参数用来控制是否采用全模式，默认为True；
      - HMM 参数用来控制是否使用 HMM 模型， 默认为True；

    - **返回**

      - results(list): 分词结果


  - ```python
    def cut_for_search(sentence, HMM=True)
    ```

    - jieba的搜索引擎模式切词，该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细

    - **参数**

      - sentence(str): 单句预测数据。
      - HMM 参数用来控制是否使用 HMM 模型， 默认为True；

    - **返回**

      - results(list): 分词结果


  - ```python
    def load_userdict(user_dict)
    ```

    - 指定自己自定义的词典，以便包含 jieba 词库里没有的词。虽然 jieba 有新词识别能力，但是自行添加新词可以保证更高的正确率。

    - **参数**

      - user_dict(str): 自定义词典的路径


  - ```python
    def extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
    ```

    - 基于 TF-IDF 算法的关键词抽取

    - **参数**

      - sentence(str): 待提取的文本
      - topK(int): 返回几个 TF/IDF 权重最大的关键词，默认值为 20
      - withWeight(bool): 为是否一并返回关键词权重值，默认值为 False
      - allowPOS(tuple): 仅包括指定词性的词，默认值为空，即不筛选

    - **返回**

      - results(list): 关键词结果


  - ```python
    def textrank(sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    ```

    - 基于 TextRank 算法的关键词抽取

    - **参数**

      - sentence(str): 待提取的文本
      - topK(int): 返回几个 TF/IDF 权重最大的关键词，默认值为 20
      - withWeight(bool): 为是否一并返回关键词权重值，默认值为 False
      - allowPOS(tuple): 仅包括指定词性的词，默认值为('ns', 'n', 'vn', 'v')

    - **返回**

      - results(list): 关键词结果


## 四、服务部署

- PaddleHub Serving可以部署一个切词服务，可以将此接口用于在线分词等web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -c serving_config.json
    ```

  - `serving_config.json`的内容如下：
    ```json
    {
      "modules_info": {
        "jieba_paddle": {
          "init_args": {
            "version": "1.0.0"
          },
          "predict_args": {}
        }
      },
      "port": 8866,
      "use_singleprocess": false,
      "workers": 2
    }
    ```

  - 这样就完成了一个切词服务化API的部署，默认端口号为8866。


- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据
    text = "今天是个好日子"

    # 设置运行配置
    # 对应本地预测jieba_paddle.cut(sentence=text, use_paddle=True)
    data = {"sentence": text, "use_paddle": True}

    # 指定预测方法为jieba_paddle并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/jieba_paddle"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  移除 fluid api

  - ```shell
    $ hub install jieba_paddle==1.0.1
    ```
