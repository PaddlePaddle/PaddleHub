# simnet_bow
|模型名称|simnet_bow|
| :--- | :---: |
|类别|文本-语义匹配|
|网络|BOW|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|245MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 模型介绍

  - 短文本语义匹配(SimilarityNet, SimNet)是一个计算短文本相似度的模型，可以根据用户输入的两个文本，计算出相似度得分。SimNet在百度各产品上广泛应用，适用于信息检索、新闻推荐、智能客服等多个应用场景，帮助企业解决语义匹配问题。该PaddleHub Module基于BOW网络结构，支持预测。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.1.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install simnet_bow
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run simnet_bow --text_1 "这道题很难" --text_2 "这道题不简单"
    ```
    - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    simnet_bow = hub.Module(name="simnet_bow")

    # Data to be predicted
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

    inputs = {"text_1": test_text_1, "text_2": test_text_2}
    results = simnet_bow.similarity(data=inputs, batch_size=2)
    print(results)

    # [{'text_1': '这道题太难了', 'text_2': '这道题是上一年的考题', 'similarity': 0.689}, {'text_1': '这道题太难了', 'text_2': '这道题不简单', 'similarity': 0.855}, {'text_1': '这道题太难了', 'text_2': '这道题很有意思', 'similarity': 0.8166}]
    ```

- ### 3、 API

  - ```python
    similarity(texts=[], use_gpu=False, batch_size=1)
    ```

    - simnet_bow预测接口，计算两个句子的cosin相似度

    - **参数**

      - texts(list): 待预测数据，第一个元素(list)为第一顺序句子，第二个元素(list)为第二顺序句子，两个元素长度相同。
        如texts=[["这道题太难了", "这道题太难了", "这道题太难了"], ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]]。
      - use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
      - batch_size(int): 批处理大小

    - **返回**

      - results(list): 带预测数据的cosin相似度

  - ```python
    get_vocab_path()
    ```
    - 获取预训练时使用的词汇表

    - **返回**

      - vocab_path(str): 词汇表路径

## 四、服务部署

- PaddleHub Serving可以部署一个在线语义匹配服务，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m simnet_bow
    ```

  - 启动时会显示加载模型过程，启动成功后显示

  - ```shell
    Loading simnet_bow successful.
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 待预测数据
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

    text = [test_text_1, test_text_2]

    # 设置运行配置
    # 对应本地预测simnet_bow.similarity(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # 指定预测方法为simnet_bow并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/simnet_bow"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  大幅提升预测性能

* 1.2.0

  模型升级，支持用于文本分类，文本匹配等各种任务迁移学习

* 1.2.1

  移除 fluid api

  - ```shell
    $ hub install simnet_bow==1.2.1
    ```
