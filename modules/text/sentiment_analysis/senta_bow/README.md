# senta_bow

|模型名称|senta_bow|
| :--- | :---: |
|类别|文本-情感分析|
|网络|BOW|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|637MB|
|最新更新日期|2021-02-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - 情感倾向分析（Sentiment Classification，简称Senta）针对带有主观描述的中文文本，可自动判断该文本的情感极性类别并给出相应的置信度，能够帮助企业理解用户消费习惯、分析热点话题和危机舆情监控，为企业提供有利的决策支持。该模型基于一个BOW结构，情感类型分为积极、消极。



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.8.0

  - paddlehub >= 1.8.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install senta_bow
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)




## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run senta_bow --input_text "这家餐厅很好吃"
    ```
  - 或者
  - ```shell
    $ hub run senta_bow --input_file test.txt
    ```  
    - test.txt 存放待预测文本， 如：
      > 这家餐厅很好吃

      > 这部电影真的很差劲

  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    senta = hub.Module(name="senta_bow")
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    results = senta.sentiment_classify(texts=test_text, use_gpu=False, batch_size=1)

    for result in results:
        print(result['text'])
        print(result['sentiment_label'])
        print(result['sentiment_key'])
        print(result['positive_probs'])
        print(result['negative_probs'])

    # 这家餐厅很好吃 1 positive 0.9782 0.0218
    # 这部电影真的很差劲 0 negative 0.0124 0.9876
    ```


- ### 3、API

  - ```python
    def sentiment_classify(texts=[], data={}, use_gpu=False, batch_size=1)
    ```
    - senta_bow预测接口，预测输入句子的情感分类(二分类，积极/消极）

    - **参数**

      - texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
      - data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
      - use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
      - batch_size(int): 批处理大小

    - **返回**

      - results(list): 情感分类结果


  - ```python
    def get_labels()
    ```

    - 获取senta_bow的类别

    - **返回**

      - labels(dict): senta_bow的类别(二分类，积极/消极)

  - ```python
    def get_vocab_path()
    ```

    - 获取预训练时使用的词汇表

    - **返回**

      - vocab_path(str): 词汇表路径




## 四、服务部署

- PaddleHub Serving可以部署一个在线情感分析服务，可以将此接口用于在线web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -m senta_bow  
    ```

  - 启动时会显示加载模型过程，启动成功后显示
    ```shell
    Loading senta_bow successful.
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据
    text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    # 设置运行配置
    # 对应本地预测senta_bow.sentiment_classify(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # 指定预测方法为senta_bow并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/senta_bow"
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

  词汇表升级

* 1.1.0

  大幅提升预测性能

* 1.2.0

  模型升级，支持用于文本分类，文本匹配等各种任务迁移学习

* 1.2.1

  移除 fluid api

  - ```shell
    $ hub install senta_bow==1.2.1
    ```
