# plato2_en_base

| 模型名称            |       plato2_en_base       |
| :------------------ | :--------------------: |
| 类别                |     文本-文本生成      |
| 网络                |  PLATO2   |
| 数据集              | 大规模开放域英文数据集 |
| 是否支持Fine-tuning |           否           |
| 模型大小            |         3.5 GB      |
| 最新更新日期        |       2022-11-05       |
| 数据指标            |           -            |

## 一、模型基本信息

- ### 模型介绍
  - PLATO2 是一个超大规模生成式对话系统模型。它承袭了 PLATO 隐变量进行回复多样化生成的特性，能够就开放域话题进行流畅深入的聊天。据公开数据，其效果超越了 Google 于 2020 年 2 月份发布的 Meena 和 Facebook AI Research 于2020 年 4 月份发布的 Blender 的效果。plato2_en_base 包含 310M 参数，可用于一键预测对话回复。由于该 Module 参数量较多，推荐使用GPU预测。


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install plato2_en_base
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```bash
    $ hub run plato2_en_base --input_text="Hello, how are you"
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="plato2_en_base")

    test_texts = ["Hello","Hello\thi, nice to meet you\tnice to meet you"]
    results = module.generate(texts=test_texts)
    for result in results:
        print(result)
    ```

- ### 3、API

  - ```python
    def generate(texts):
    ```

    - 预测API，输入对话上下文，输出机器回复。
    - **参数**
      - texts (list\[str\] or str): 如果不在交互模式中，texts应为list，每个元素为一次对话的上下文，上下文应包含人类和机器人的对话内容，不同角色之间的聊天用分隔符"\t"进行分割；例如[["Hello\thi, nice to meet you\tnice to meet you"]]。这个输入中包含1次对话，机器人回复了"hi, nice to meet you"后人类回复“nice to meet you”，现在轮到机器人回复了。如果在交互模式中，texts应为str，模型将自动构建它的上下文。

  - ```python
    def interactive_mode(max_turn=6):
    ```

    - 进入交互模式。交互模式中，generate接口的texts将支持字符串类型。
    - **参数**
      - max_turn (int): 模型能记忆的对话轮次，当max_turn = 1时，模型只能记住当前对话，无法获知之前的对话内容。


## 四、服务部署

- PaddleHub Serving可以部署一个在线对话机器人服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m plato2_en_base -p 8866
    ```

  - 这样就完成了一个对话机器人服务化API的部署，默认端口号为8866。
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    data = {'texts':["Hello","Hello\thi, nice to meet you\tnice to meet you"]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/plato2_en_base"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 保存结果
    results = r.json()["results"]
    for result in results:
        print(result)
    ```

  - 关于PaddleHub Serving更多信息参考[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布
  
* 1.1.0

  移除 Fluid API

  - ```shell
    $ hub install plato2_en_base==1.1.0
    ```
