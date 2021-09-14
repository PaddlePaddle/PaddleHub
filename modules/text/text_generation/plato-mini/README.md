# plato-mini

| 模型名称            |       plato-mini       |
| :------------------ | :--------------------: |
| 类别                |     文本-文本生成      |
| 网络                |  Unified Transformer   |
| 数据集              | 十亿级别的中文对话数据 |
| 是否支持Fine-tuning |           否           |
| 模型大小            |         5.28K          |
| 最新更新日期        |       2021-06-30       |
| 数据指标            |           -            |

## 一、模型基本信息

- ### 模型介绍
  - [UnifiedTransformer](https://arxiv.org/abs/2006.16779)以[Transformer](https://arxiv.org/abs/1706.03762) 编码器为网络基本组件，采用灵活的注意力机制，十分适合文本生成任务，并在模型输入中加入了标识不同对话技能的special token，使得模型能同时支持闲聊对话、推荐对话和知识对话。
该模型在十亿级别的中文对话数据上进行预训练，通过PaddleHub加载后可直接用于对话任务，仅支持中文对话。


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)
  
- ### 2、安装

  - ```shell
    $ hub install plato-mini
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- plato-mini不支持一行预测，仅支持python代码预测

- ### 1、预测代码示例

  - ```python
    # 非交互模式
    import paddlehub as hub
    
    model = hub.Module(name='plato-mini')
    data = [["你是谁？"], ["你好啊。", "吃饭了吗？",]]
    result = model.predict(data)
    print(result)
    
    # ['我是一个小角色,我是在玩游戏', '吃过了呢,你吃了没?']
    # 每次的运行结果可能有所不同
    ```
    
  - ```python
    # 交互模式
    # 使用命令行与机器人对话
    import paddlehub as hub
    import readline
    
    model = hub.Module(name='plato-mini')
    with model.interactive_mode(max_turn=3):
        while True:
            human_utterance = input("[Human]: ").strip()
            robot_utterance = model.predict(human_utterance)[0]
            print("[Bot]: %s"%robot_utterance)
    ```

- ### 2、API

  - ```python
    def predict(data, max_seq_len=512, batch_size=1, use_gpu=False, **kwargs):
    ```

    - 预测API，输入对话上下文，输出机器回复。
    - **参数**
      - data(Union[List[List[str](https://www.paddlepaddle.org.cn/hubdetail?name=plato-mini&en_category=TextGeneration)], str]): 在非交互模式中，数据类型为List[List[str](https://www.paddlepaddle.org.cn/hubdetail?name=plato-mini&en_category=TextGeneration)]，每个样本是一个List[str](https://www.paddlepaddle.org.cn/hubdetail?name=plato-mini&en_category=TextGeneration)，表示为对话内容
      - max_seq_len(int): 每个样本的最大文本长度
      - batch_size(int): 进行预测的batch_size
      - use_gpu(bool): 是否使用gpu执行预测
      - kwargs: 预测时传给模型的额外参数，以keyword方式传递。其余的参数详情请查看[UnifiedTransformer](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/dialogue/unified_transformer)。
    - **返回**
      - results(List[str](https://www.paddlepaddle.org.cn/hubdetail?name=plato-mini&en_category=TextGeneration)): 每个元素为相应对话中模型的新回复
    
  - ```python
    def interactive_mode(max_turn=3):
    ```
  
    - 配置交互模式并进入。
    - **参数**
      - max_turn(int): 模型能记忆的对话轮次，当max_turn为1时，模型只能记住当前对话，无法获知之前的对话内容。


## 四、服务部署

- PaddleHub Serving可以部署一个在线对话机器人服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m plato-mini -p 8866
    ```

  - 这样就完成了一个对话机器人服务化API的部署，默认端口号为8866。
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    
    texts = [["今天是个好日子"], ["天气预报说今天要下雨"]]
    data = {"data": texts}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/plato_mini"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}
    
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    
    # {'msg': '', 'results': ['是个好日子啊!', '下雨就不出门了,在家宅着吧'], 'status': '000'}
    # 每次的运行结果可能有所不同
    ```
    
  - 关于PaddleHub Serving更多信息参考[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布
  
  - ```shell
    $ hub install plato-mini==1.0.0
    ```
