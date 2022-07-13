# lac

|模型名称|lac|
| :--- | :---: |
|类别|文本-词法分析|
|网络|BiGRU+CRF|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|35MB|
|最新更新日期|2021-02-26|
|数据指标|Precision=88.0%，Recall=88.7%，F1-Score=88.4%|



## 一、模型基本信息

- ### 模型介绍

  - Lexical Analysis of Chinese，简称 LAC，是一个联合的词法分析模型，能整体性地完成中文分词、词性标注、专名识别任务。在百度自建数据集上评测，LAC效果：Precision=88.0%，Recall=88.7%，F1-Score=88.4%。该PaddleHub Module支持预测。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/130606395-60691079-d33f-4d74-a980-b4d9e3bc663e.png"   height = "300" hspace='10'/> <br />
</p>

  - 更多详情请参考：[LAC论文](https://arxiv.org/abs/1807.01882)




## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2

  - paddlehub >= 1.6.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

  - 若使用词典干预功能，额外依赖第三方库 pyahocorasick

  - ```shell
    $ pip install pyahocorasick
    ```

- ### 2、安装

  - ```shell
    $ hub install lac
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)




## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run lac --input_text "今天是个好日子"
    ```
  - 或者
  - ```shell
    $ hub run lac --input_file test.txt --user_dict user.dict
    ```

    - test.txt 存放待分词文本， 如：
      - ```shell
        今天是个好日子  
        今天天气晴朗
        ```
    - user.dict 为用户自定义词典，可以不指定，当指定自定义词典时，可以干预默认分词结果。如：
      - ```shell
        春天/SEASON
        花/n 开/v
        秋天的风
        落 阳  
        ```
      - 词典文件每行表示一个定制化的item，由一个单词或多个连续的单词组成，每个单词后使用'/'表示标签，如果没有'/'标签则会使用模型默认的标签。每个item单词数越多，干预效果会越精准。

    - Note：该PaddleHub Module使用词典干预功能时，依赖于第三方库pyahocorasick，请自行安装

  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]

    results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)

    for result in results:
        print(result['word'])
        print(result['tag'])

    # ['今天', '是', '个', '好日子']
    # ['TIME', 'v', 'q', 'n']
    # ['天气预报', '说', '今天', '要', '下雨']
    # ['n', 'v', 'TIME', 'v', 'v']
    # ['下', '一班', '地铁', '马上', '就要', '到', '了']
    # ['f', 'm', 'n', 'd', 'v', 'v', 'xc']
    ```



- ### 3、API

  - ```python
    def __init__(user_dict=None)
    ```
    - 构造LAC对象

    - **参数**

      - user_dict(str): 自定义词典路径。如果需要使用自定义词典，则可通过该参数设置，否则不用传入该参数。


  - ```python
    def cut(text, use_gpu=False, batch_size=1, return_tag=True)
    ```

    - lac预测接口，预测输入句子的分词结果

    - **参数**

      - text(str or list): 待预测数据，单句预测数据（str类型）或者批量预测（list，每个元素为str
      - use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
      - batch_size(int): 批处理大小
      - return_tag(bool): 预测结果是否需要返回分词标签结果


  - ```python
    def lexical_analysis(texts=[], data={}, use_gpu=False, batch_size=1, user_dict=None, return_tag=True)
    ```

    - **该接口将会在未来版本被废弃，如有需要，请使用cut接口预测**

    - lac预测接口，预测输入句子的分词结果

    - **参数**

      - texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
      - data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃
      - use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
      - batch_size(int): 批处理大小
      - return_tag(bool): 预测结果是否需要返回分词标签结果

    - **返回**

      - results(list): 分词结果


  - ```python
    def set_user_dict(dict_path)
    ```

    - 加载用户自定义词典

    - **参数**

      - dict_path(str ): 自定义词典路径


  - ```python
    def del_user_dict()
    ```

    - 删除自定义词典


  - ```python
    def get_tags()
    ```

    - 获取lac的标签

    - **返回**

      - tag_name_dict(dict): lac的标签




## 四、服务部署

- PaddleHub Serving可以部署一个在线词法分析服务，可以将此接口用于词法分析、在线分词等在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -c serving_config.json
    ```

    `serving_config.json`的内容如下：
    ```json
    {
      "modules_info": {
        "lac": {
          "init_args": {
            "version": "2.1.0",
            "user_dict": "./test_dict.txt"
          },
          "predict_args": {}
        }
      },
      "port": 8866,
      "use_singleprocess": false,
      "workers": 2
    }
    ```
  - 其中user_dict含义为自定义词典路径，如果不使用lac自定义词典功能，则可以不填入。

  - 这样就完成了一个词法分析服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    # 待预测数据
    text = ["今天是个好日子", "天气预报说今天要下雨"]

    # 设置运行配置
    # 对应本地预测lac.analysis_lexical(texts=text, batch_size=1, use_gpu=True)
    data = {"texts": text, "batch_size": 1, "use_gpu":True}

    # 指定预测方法为lac并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/lac"
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

  修复python2中编码问题

* 1.1.0

  支持词典干预，通过配置自定义词典可以对LAC默认分词结果进行干预

* 1.1.1

  修复输入文本中带有/字符时，使用词典干预会崩溃的问题

* 2.0.0

  修复输入文本为空、“ ”或者“\n”，使用LAC会崩溃的问题
  更新embedding_size, hidden_size为128，压缩模型，性能提升

* 2.1.0

  lac预测性能大幅提升
  支持是否返回分词标签tag，同时简化预测接口使用

* 2.1.1

  当输入文本为空字符串“”，返回切词后结果为空字符串“”，分词tag也为“”

* 2.2.0

  升级自定义词典功能，支持增加不属于lac默认提供的词性

* 2.2.1

  移除 fluid api

  - ```shell
    $ hub install lac==2.2.1
    ```
