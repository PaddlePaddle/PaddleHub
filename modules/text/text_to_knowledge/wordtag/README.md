# WordTag

|模型名称|WordTag|
| :--- | :---: | 
|类别|文本-文本知识关联|
|网络|ERNIE-CTM+CRF|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|549MB|
|最新更新日期|2021-10-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - WordTag（中文词类知识标注工具）是首个能够覆盖所有中文词汇的词类知识标注工具，旨在为中文文本解析提供全面、丰富的知识标注结果，可以应用于模板（挖掘模板、解析模板）生成与匹配、知识挖掘(新词发现、关系挖掘)等自然语言处理任务中，提升文本解析与挖掘精度；也可以作为中文文本特征生成器，为各类机器学习模型提供文本特征。

  <p align="center">
  <img src="https://user-images.githubusercontent.com/40840292/137913774-186f72e9-c51b-469e-8356-b72bafc4d926.png" hspace='10'/> <br />
  </p>

  - WordTag特点

    - 覆盖所有中文词汇的词类体系，更丰富的知识标注结果
      - WordTag使用的词类体系为覆盖所有中文词汇的词类体系，包括各类实体词与非实体词（如概念、实体/专名、语法词等）。WordTag开源版对部分类目（如组织机构等），做了更细类目的划分识别（如，医疗卫生机构、体育组织机构），对仅使用文本信息难以细分的类目（如人物类、作品类、品牌名等），不做更细粒度的词类识别。用户需要细粒度的词类识别时，可利用百科知识树的类别体系自行定制。
    
    - 整合百科知识树链接结果，获得更丰富的标注知识
      - 如上图示例所示，各个切分标注结果中，除词类标注外，还整合了百科知识树的链接结果，用户可以结合百科知识树数据共同使用：如，利用百科知识树中的subtype获得更细的上位粒度，利用term的百科信息获得更加丰富的知识等。

    - 可定制的词类序列标注框架
      - WordTag开源版标注使用的词类体系是我们在实践中对百科文本解析应用较好的一个版本，不同类型文本（如，搜索query、新闻资讯）的词类分布不同，用户可以利用百科知识树定制自己的词类体系和训练样本，构建自己的WordTag应用版，以获得更好的适配效果。例如，可将自定义的词表按照百科知识树的字段定义好，挂接/整合到百科知识树上，即可使用自己的Term数据定制标注样本和标注任务。

  - 模型结构
    - 模型使用ERNIE-CTM+CRF训练而成，预测时使用viterbi解码，模型结构如下：

  <p align="center">
  <img src="https://user-images.githubusercontent.com/40840292/137915351-0ef2609c-1aab-4c7e-8634-8726f8ddb30d.png" hspace='10'/> <br />
  </p>

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0
  
  - paddlenlp >= 2.1.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install wordtag
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run wordtag --input_text="《孤女》是2010年九州出版社出版的小说，作者是余兼羽。"
    ```
  - 通过命令行方式实现WordTag模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    # Load WordTag
    module = hub.Module(name="wordtag")

    # String input
    results = module.predict("《孤女》是2010年九州出版社出版的小说，作者是余兼羽。")
    print(results)
    # [{'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]

    # List input
    results = module.predict(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
    print(results)
    # [{'text': '热梅茶是一道以梅子为主要原料制作的茶饮', 'items': [{'item': '热梅茶', 'offset': 0, 'wordtag_label': '饮食类_饮品', 'length': 3}, {'item': '是', 'offset': 3, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '一道', 'offset': 4, 'wordtag_label': '数量词', 'length': 2}, {'item': '以', 'offset': 6, 'wordtag_label': '介词', 'length': 1, 'termid': '介词_cb_以'}, {'item': '梅子', 'offset': 7, 'wordtag_label': '饮食类', 'length': 2, 'termid': '饮食_cb_梅'}, {'item': '为', 'offset': 9, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_为'}, {'item': '主要原料', 'offset': 10, 'wordtag_label': '物体类', 'length': 4, 'termid': '物品_cb_主要原料'}, {'item': '制作', 'offset': 14, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_制作'}, {'item': '的', 'offset': 16, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '茶饮', 'offset': 17, 'wordtag_label': '饮食类_饮品', 'length': 2, 'termid': '饮品_cb_茶饮'}]}, {'text': '《孤女》是2010年九州出版社出版的小说，作者是余兼羽', 'items': [{'item': '《', 'offset': 0, 'wordtag_label': 'w', 'length': 1}, {'item': '孤女', 'offset': 1, 'wordtag_label': '作品类_实体', 'length': 2}, {'item': '》', 'offset': 3, 'wordtag_label': 'w', 'length': 1}, {'item': '是', 'offset': 4, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '2010年', 'offset': 5, 'wordtag_label': '时间类', 'length': 5, 'termid': '时间阶段_cb_2010年'}, {'item': '九州出版社', 'offset': 10, 'wordtag_label': '组织机构类', 'length': 5, 'termid': '组织机构_eb_九州出版社'}, {'item': '出版', 'offset': 15, 'wordtag_label': '场景事件', 'length': 2, 'termid': '场景事件_cb_出版'}, {'item': '的', 'offset': 17, 'wordtag_label': '助词', 'length': 1, 'termid': '助词_cb_的'}, {'item': '小说', 'offset': 18, 'wordtag_label': '作品类_概念', 'length': 2, 'termid': '小说_cb_小说'}, {'item': '，', 'offset': 20, 'wordtag_label': 'w', 'length': 1}, {'item': '作者', 'offset': 21, 'wordtag_label': '人物类_概念', 'length': 2, 'termid': '人物_cb_作者'}, {'item': '是', 'offset': 23, 'wordtag_label': '肯定词', 'length': 1, 'termid': '肯定否定词_cb_是'}, {'item': '余兼羽', 'offset': 24, 'wordtag_label': '人物类_实体', 'length': 3}]}]
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
      - linking(bool): 是否返回百科知识树的链接结果，默认为True。

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
                'items': list\[dict\], 标注结果, 包含以下信息：
                  {
                    'item': str, 分词结果。
                    'offset': int, 与输入文本首个字的偏移值。
                    'wordtag_label': str, 词类知识标注结果。
                    'length': int, 词汇长度。
                    'termid': str, 与百科知识树的链接结果。
                  }
            }

## 四、服务部署

- PaddleHub Serving可以部署一个在线中文词类知识标注服务，可以将此接口用于在线web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -m wordtag
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据(input string)
    text = ["《孤女》是2010年九州出版社出版的小说，作者是余兼羽。"]

    # 设置运行配置
    data = {"texts": text}
    
    # 指定预测方法为WordTag并发送post请求，content-type类型应指定json方式
    url = "http://127.0.0.1:8866/predict/wordtag"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())

    # 待预测数据(input list)
    text = ["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"]

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
    $ hub install wordtag==1.0.0
    ```
