# NPTag

|模型名称|NPTag|
| :--- | :---: | 
|类别|文本-文本知识关联|
|网络|ERNIE-CTM|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|378MB|
|最新更新日期|2021-12-10|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - NPTag（名词短语标注工具）是首个能够覆盖所有中文名词性词汇及短语的细粒度知识标注工具，旨在解决NLP中，名词性短语收录不足，导致的OOV（out-of-vocabulary，超出收录词表）问题。可直接应用构造知识特征，辅助NLP任务

  - NPTag特点

    - 包含2000+细粒度类别，覆盖所有中文名词性短语的词类体系，更丰富的知识标注结果
        - NPTag试用的词类体系未覆盖所有中文名词性短语的词类体系，对所有类目做了更细类目的识别（如注射剂、鱼类、博物馆等），共包含2000+细粒度类别，且可以直接关联百科知识树。
    - 可自由定制的分类框架
        - NPTag开源版标注使用的词类体系是我们在实践中对**百科词条**分类应用较好的一个版本，用户可以自由定制自己的词类体系和训练样本，构建自己的NPTag，以获得更好的适配效果。例如，可按照自定义的类别构造训练样本，使用小学习率、短训练周期微调NPTag模型，即可获得自己定制的NPTag工具。

  - 模型结构
    - NPTag使用ERNIE-CTM+prompt训练而成，使用启发式搜索解码，保证分类结果都在标签体系之内。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0
  
  - paddlenlp >= 2.2.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install nptag
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run nptag --input_text="糖醋排骨"
    ```
  - 通过命令行方式实现NPTag模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    # Load NPTag
    module = hub.Module(name="nptag")

    # String input
    results = module.predict("糖醋排骨")
    print(results)
    # [{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}]

    # List input
    results = module.predict(["糖醋排骨", "红曲霉菌"])
    print(results)
    # [{'text': '糖醋排骨', 'label': '菜品', 'category': '饮食类_菜品'}, {'text': '红曲霉菌', 'label': '微生物', 'category': '生物类_微生物'}]
    ```
    
- ### 3、API

  - ```python
    def __init__(
      batch_size=32,
      max_seq_length=128,
      linking=True,
    )
    ```

    - **参数**

      - batch_size(int): 每个预测批次的样本数目，默认为32。
      - max_seq_length(int): 最大句子长度，默认为128。
      - linking(bool): 实现与WordTag类别标签的linking，默认为True。

  - ```python
    def predict(texts)
    ```
    - 预测接口，输入文本，输出词类标注结果以及百科知识树的链接结果。

    - **参数**

      - texts(str or list\[str\]): 待预测数据。

    - **返回**

      - results(list\[dict\]): 输出结果。每个元素都是dict类型，包含以下信息：  
     
            {
                'text': str, 原始文本。
                'label': str，预测结果。
                'category'：str，对应的WordTag类别标签。
            }

## 四、服务部署

- PaddleHub Serving可以部署一个在线中文名词短语标注服务，可以将此接口用于在线web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -m nptag
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据(input string)
    text = ["糖醋排骨"]

    # 设置运行配置
    data = {"texts": text}
    
    # 指定预测方法为WordTag并发送post请求，content-type类型应指定json方式
    url = "http://127.0.0.1:8866/predict/nptag"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())

    # 待预测数据(input list)
    text = ["糖醋排骨", "红曲霉菌"]

    # 设置运行配置
    data = {"texts": text}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install nptag==1.0.0
    ```
