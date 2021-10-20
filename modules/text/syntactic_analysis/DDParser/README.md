# DDParser

|模型名称|DDParser|
| :--- | :---: | 
|类别|文本-句法分析|
|网络|Deep Biaffine Attention|
|数据集|搜索query、网页文本、语音输入等数据|
|是否支持Fine-tuning|否|
|模型大小|61MB|
|最新更新日期|2021-10-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - DDParser（Baidu Dependency Parser）是百度NLP基于大规模标注数据和深度学习平台飞桨研发的中文依存句法分析工具，可帮助用户直接获取输入文本中的关联词对、长距离依赖词对等。
  


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0
  
  - paddlenlp >= 2.1.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ddparser
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run ddparser --input_text="百度是一家高科技公司"
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub

    # Load ddparser
    module = hub.Module(name="ddparser")

    # String input
    results = module.parse("百度是一家高科技公司")
    print(results)
    # [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}]

    # List input
    results = module.parse(["百度是一家高科技公司", "他送了一本书"])
    print(results)
    # [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB']}, {'word': ['他', '送', '了', '一本', '书'], 'head': [2, 0, 2, 5, 2], 'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB']}]

    # Use POS Tag and probability
    module = hub.Module(name="ddparser", prob=True, use_pos=True)
    results = module.parse("百度是一家高科技公司")
    print(results)
    # [{'word': ['百度', '是', '一家', '高科技', '公司'], 'head': [2, 0, 5, 5, 2], 'deprel': ['SBV', 'HED', 'ATT', 'ATT', 'VOB'], 'postag': ['ORG', 'v', 'm', 'n', 'n'], 'prob': [1.0, 1.0, 1.0, 1.0, 1.0]}]

    # Visualization mode
    module = hub.Module(name="ddparser", return_visual=True)
    data = module.visualize("百度是一家高科技公司")
    cv2.imwrite('test.jpg', data)
    ```
    
- ### 3、API

  - ```python
    def __init__(
      tree=True,
      prob=False,
      use_pos=False,
      batch_size=1,
      return_visual=False)
    ```
    - 模块初始化。
    
    - **参数**

      - tree(bool): 输出结果是否需要满足树状结构，默认为True。
      - prob(bool): 是否输出概率值，默认为False。
      - use_pos(bool): 是否输出词性标签，默认为False。
      - batch_size(int): 批大小，默认为1。
      - return_visual(bool): 是否返回可视化结果（需配合visualize api使用），默认为False。

  - ```python
    def parse(texts)
    ```
    - 依存分析接口，输入文本，输出依存关系。

    - **参数**

      - texts(str or list\[str\]]): 待预测数据。

    - **返回**

      - results(list\[dict\]): 依存分析结果。每个元素都是dict类型，包含以下信息：  
     
            {
                'word': list[str], 分词结果。
                'head': list[int], 当前成分其支配者的id。
                'deprel': list[str], 当前成分与支配者的依存关系。
                'prob': list[float], 从属者和支配者依存的概率。
                'postag': list[str], 词性标签，只有当texts的元素是未分词的字符串时包含这个键。
                'visual': numpy.ndarray, 图像数组，可以使用cv2.imshow显示图像或cv2.imwrite保存图像。
            }
      

  - ```python
    def visualize(text)
    ```

    - 可视化接口，输入文本信息，输出依存图形数组。

    - **参数**

      - text(str): 输入文本，支持string格式的单条文本输入。

    - **返回**

      - data(numpy.ndarray): 图像数组。可以使用cv2.imshow显示图像或cv2.imwrite保存图像。



## 四、服务部署

- PaddleHub Serving可以部署一个在线句法分析服务，可以将此接口用于在线web应用。

- ## 第一步：启动PaddleHub Serving

  - 运行启动命令：
    ```shell
    $ hub serving start -m ddparser
    ```

  - 启动时会显示加载模型过程，启动成功后显示
    ```shell
    Loading ddparser successful.
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json

    # 待预测数据(input string)
    text = ["百度是一家高科技公司"]

    # 设置运行配置
    data = {"texts": text}
    
    # 指定预测方法为DuDepParser并发送post请求，content-type类型应指定json方式
    url = "http://127.0.0.1:8866/predict/ddparser"
    headers = {"Content-Type": "application/json"}
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    print(r.json())
    # {'msg': '', 'results': [{'deprel': ['SBV', 'HED', 'ATT', 'VOB'], 'head': ['2', '0', '4', '2'], 'word': ['百度', '是', '一家', '公司']}], 'status': '000'}

    # 待预测数据(input list)
    text = ["百度是一家公司", "他送了一本书"]

    # 设置运行配置
    data = {"texts": text}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    # {'msg': '', 'results': [{'deprel': ['SBV', 'HED', 'ATT', 'VOB'], 'head': ['2', '0', '4', '2'], 'word': ['百度', '是', '一家', '公司']}, {'deprel': ['SBV', 'HED', 'MT', 'ATT', 'VOB'], 'head': ['2', '0', '2', '5', '2'], 'word': ['他', '送', '了', '一本', '书']}], 'status': '000'}
    
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)


## 五、更新历史

* 1.0.0

  初始发布

* 1.1.0

  适配paddlepaddle 2.1版本

  - ```shell
    $ hub install ddparser==1.1.0
    ```
