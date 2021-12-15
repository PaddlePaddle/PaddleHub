# ERNIE-CSC

|模型名称|ERNIE-CSC|
| :--- | :---: | 
|类别|文本-文本纠错|
|网络|ERNIE-CSC|
|数据集|SIGHAN|
|是否支持Fine-tuning|否|
|模型大小|436MB|
|最新更新日期|2021-12-10|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - 中文文本纠错任务是一项NLP基础任务，其输入是一个可能含有语法错误的中文句子，输出是一个正确的中文句子。语法错误类型很多，有多字、少字、错别字等，目前最常见的错误类型是错别字。大部分研究工作围绕错别字这一类型进行研究。本文实现了百度在ACL 2021上提出结合拼音特征的Softmask策略的中文错别字纠错的下游任务网络，并提供预训练模型，模型结构如下：

  <p align="center">
  <img src="https://user-images.githubusercontent.com/40840292/146150468-9168651a-1fa0-4d60-9871-69e494d1d370.png" hspace='10'/> <br />
  </p>

  - 更多详情请[参考论文](https://aclanthology.org/2021.findings-acl.198.pdf)

  - 注：论文中暂未开源融合字音特征的预训练模型参数(即MLM-phonetics)，所以本文提供的纠错模型是在ERNIE-1.0的参数上进行Finetune，纠错模型结构与论文保持一致。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0
  
  - paddlenlp >= 2.2.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ernie-csc
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run ernie-csc --input_text="遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。"
    ```
  - 通过命令行方式实现文本纠错ernie-csc模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    # Load ernie-csc
    module = hub.Module(name="ernie-csc")

    # String input
    results = module.predict("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")
    print(results)
    # [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}]

    # List input
    results = module.predict(['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。'])
    print(results)
    # [{'source': '遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'target': '遇到逆境时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', 'errors': [{'position': 3, 'correction': {'竟': '境'}}]}, {'source': '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。', 'target': '人生就是如此，经过磨练才能让自己更加茁壮，才能使自己更加乐观。', 'errors': [{'position': 18, 'correction': {'拙': '茁'}}]}]
    ```
    
- ### 3、API

  - ```python
    def __init__(batch_size=32)
    ```

    - **参数**

      - batch_size(int): 每个预测批次的样本数目，默认为32。

  - ```python
    def predict(texts)
    ```
    - 预测接口，输入文本，输出文本纠错结果。

    - **参数**

      - texts(str or list\[str\]): 待预测数据。

    - **返回**

      - results(list\[dict\]): 输出结果。每个元素都是dict类型，包含以下信息：  

            {
                'source': str, 输入文本。
                'target': str, 模型预测结果。
                'errors': list[dict], 错误字符的详细信息，包含如下信息:
                    {
                        'position': int, 错误字符的位置。
                        'correction': dict, 错误字符及其对应的校正结果。
                    }
            }


## 四、服务部署

- PaddleHub Serving可以部署一个在线文本纠错服务，可以将此接口用于在线web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -m ernie-csc
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据(input string)
    text = ["遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。"]

    # 设置运行配置
    data = {"texts": text}
    
    # 指定预测方法为ernie-csc并发送post请求，content-type类型应指定json方式
    url = "http://127.0.0.1:8866/predict/ernie-csc"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())

    # 待预测数据(input list)
    text = ['遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。', '人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。']

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
    $ hub install ernie-csc==1.0.0
    ```
