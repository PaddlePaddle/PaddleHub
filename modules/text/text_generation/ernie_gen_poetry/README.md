# ernie_gen_poetry

| 模型名称            | ernie_gen_poetry |
| :------------------ | :--------------: |
| 类别                |  文本-文本生成   |
| 网络                |    ERNIE-GEN     |
| 数据集              |  开源诗歌数据集  |
| 是否支持Fine-tuning |        否        |
| 模型大小            |       422M       |
| 最新更新日期        |    2021-02-26    |
| 数据指标            |        -         |

## 一、模型基本信息

- 模型介绍
  - ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。
  - ernie_gen_poetry采用开源诗歌数据集进行微调，可用于生成诗歌。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133191670-8eb1c542-f8e8-4715-adb2-6346b976fab1.png"  width="600" hspace='10'/>
</p>

- 更多详情请参考：[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)


## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)
  - paddlenlp >= 2.0.0

- ### 2、安装

  - ```shell
    $ hub install ernie_gen_poetry
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run ernie_gen_poetry --input_text="昔年旅南服，始识王荆州。" --use_gpu True --beam_width 5
    ```
    
    - input_text: 诗歌的开头。
    - use_gpu: 是否采用GPU进行预测。
    - beam_width: beam search宽度，决定每个诗歌开头输出的下文数目。
    
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    
    module = hub.Module(name="ernie_gen_poetry")
    
    test_texts = ['昔年旅南服，始识王荆州。', '高名出汉阴，禅阁跨香岑。']
    results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
    for result in results:
        print(result)
    
    # ['一见便倾盖，论交更绸缪。别来二十年，日月如奔流。人生会合难，俯仰成春秋。', '一见便倾盖，论交更绸缪。别来二十年，日月如奔流。人生会合难，况乃岁月遒。君家富文史，我老无田畴。相逢不相识，各在天一陬。人生百年内，聚散如浮沤。况我与夫子，相逢', '一见便倾盖，论交更绸缪。别来二十年，日月如奔流。人生会合难，况乃岁月遒。君家富文史，我老无田畴。相逢不相识，各在天一陬。人生百年内，聚散如浮沤。况我与君别，飘零', '一见便倾盖，论交更绸缪。别来二十年，日月如奔流。人生会合难，况乃岁月遒。君家富文史，我老无田畴。相逢不相识，各在天一陬。人生百年内，聚散如浮沤。况复各异乡，各在', '一见便倾盖，论交更绸缪。别来二十年，日月如奔流。人生会合难，况乃岁月遒。君家富文史，我老无田畴。相逢不相识，各在天一陬。人生百年内，聚散如浮沤。况复各异乡，风雨']
    # ['地僻无尘到，山高见水深。钟声传远寺，塔影落前林。欲问西来意，庭前柏树林。', '地僻无尘到，山高见水深。钟声传远寺，塔影落前林。欲问西来意，庭前柏树阴。', '地僻无尘到，山高见水深。钟声传远寺，塔影落前林。欲问西来意，庭前有桂林。', '地僻无尘到，山高见水深。钟声传远寺，塔影落前林。欲问西来意，庭前柏正森。', '地僻无尘到，山高见水深。钟声传远寺，塔影落前林。欲问西来意，庭前有桂阴。']
    ```

- ### 3、API

  - ```python
    def generate(texts, use_gpu=False, beam_width=5):
    ```

    - 预测API，输入诗歌开头，输出诗歌下文。
    - **参数**
      - texts (list[str]): 诗歌的开头；
      - use_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**；
      - beam_width: beam search宽度，决定每个诗歌开头输出的下文数目。
    - **返回**
      - results (list[list]\[str]): 诗歌下文，每个诗歌开头会生成beam_width个下文。


## 四、服务部署

- PaddleHub Serving可以部署一个在线诗歌生成服务，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m ernie_gen_poetry -p 8866
    ```

  - 这样就完成了一个服务化API的部署，默认端口号为8866。
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    
    # 发送HTTP请求
    
    data = {'texts':['昔年旅南服，始识王荆州。', '高名出汉阴，禅阁跨香岑。'],
            'use_gpu':False, 'beam_width':5}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/ernie_gen_poetry"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    # 保存结果
    results = r.json()["results"]
    for result in results:
        print(result)
    
    # serving运行结果同本地运行结果（见上）
    ```
    
  - 关于PaddleHub Serving更多信息参考[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  修复windows中的编码问题

* 1.0.2

  完善API的输入文本检查

- 1.1.0

  修复兼容性问题

  - ```shell
    $ hub install ernie_gen_poetry==1.1.0
    ```
