# porn_detection_lstm

| 模型名称            |  senta_bilstm  |
| :------------------ | :------------: |
| 类别                | 文本-文本审核  |
| 网络                |      LSTM      |
| 数据集              | 百度自建数据集 |
| 是否支持Fine-tuning |       否       |
| 模型大小            |       1M       |
| 最新更新日期        |   2021-02-26   |
| 数据指标            |       -        |

## 一、模型基本信息

- ### 模型介绍
  - 色情检测模型可自动判别文本是否涉黄并给出相应的置信度，对文本中的色情描述、低俗交友、污秽文案进行识别。
  - porn_detection_lstm采用LSTM网络结构并按字粒度进行切词，具有较高的分类精度。该模型最大句子长度为256字，仅支持预测。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 1.6.2

  - paddlehub >= 1.6.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install porn_detection_lstm
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run porn_detection_lstm --input_text "黄片下载"
    ```

  - 或者

  - ```shell
    $ hub run porn_detection_lstm --input_file test.txt
    ```

    - 其中test.txt存放待审查文本，每行仅放置一段待审核文本

  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    porn_detection_lstm = hub.Module(name="porn_detection_lstm")

    test_text = ["黄片下载", "打击黄牛党"]

    results = porn_detection_lstm.detection(texts=test_text, use_gpu=True, batch_size=1)

    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        print(results[index])

    # 输出结果如下：
    # {'text': '黄片下载', 'porn_detection_label': 1, 'porn_detection_key': 'porn', 'porn_probs': 0.9879, 'not_porn_probs': 0.0121}
    # {'text': '打击黄牛党', 'porn_detection_label': 0, 'porn_detection_key': 'not_porn', 'porn_probs': 0.0004, 'not_porn_probs': 0.9996}
    ```

- ### 3、API

  - ```python
    def detection(texts=[], data={}, use_gpu=False, batch_size=1):
    ```

    - porn_detection_lstm预测接口，鉴定输入句子是否为黄文

    - **参数**
      - texts(list[str]): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
      - data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
      - use_gpu(bool): 是否使用GPU预测
      - batch_size(int): 批处理大小

    - **返回**
      - results(list): 鉴定结果

  - ```python
    def get_labels():
    ```
    - 获取porn_detection_lstm的可识别的类别及其编号

    - **返回**
      - labels(dict): porn_detection_lstm的类别及其对应编号(二分类，是/不是)

  - ```python
    def get_vocab_path():
    ```
    - 获取预训练时使用的词汇表

    - **返回**
      - vocab_path(str): 词汇表路径

## 四、服务部署

- PaddleHub Serving可以部署一个在线色情文案检测服务，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m porn_detection_lstm -p 8866
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 待预测数据
    text = ["黄片下载", "打击黄牛党"]

    # 设置运行配置
    # 对应本地预测porn_detection_lstm.detection(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # 指定预测方法为porn_detection_lstm并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/porn_detection_lstm"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(json.dumps(r.json(), indent=4, ensure_ascii=False))
    ```

  - 关于PaddleHub Serving更多信息参考[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  大幅提升预测性能，同时简化接口使用

* 1.1.1

  移除 fluid api

  - ```shell
    $ hub install porn_detection_lstm==1.1.1
    ```
