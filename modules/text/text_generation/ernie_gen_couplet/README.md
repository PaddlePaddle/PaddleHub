# ernie_gen_couplet

| 模型名称            | ernie_gen_couplet |
| :------------------ | :---------------: |
| 类别                |   文本-文本生成   |
| 网络                |     ERNIE-GEN     |
| 数据集              |  开源对联数据集   |
| 是否支持Fine-tuning |        否         |
| 模型大小            |       421M        |
| 最新更新日期        |    2021-02-26     |
| 数据指标            |         -         |

## 一、模型基本信息

- 模型介绍
  - ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。
  - ernie_gen_couplet采用开源对联数据集进行微调，输入上联，可生成下联。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133191670-8eb1c542-f8e8-4715-adb2-6346b976fab1.png"  width="600" hspace='10'/>
</p>

- 更多详情请参考：
  - 论文：[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)
  - 代码：https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-gen/

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0
  - paddlehub >= 2.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)
  - paddlenlp >= 2.0.0

- ### 2、安装

  - ```shell
    $ hub install ernie_gen_couplet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run ernie_gen_couplet --input_text="人增福寿年增岁" --use_gpu True --beam_width 5
    ```
    
    - input_text: 上联文本
    - use_gpu: 是否采用GPU进行预测
    - beam_width: beam search宽度，决定每个上联输出的下联数量。
    
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    
    module = hub.Module(name="ernie_gen_couplet")
    
    test_texts = ["人增福寿年增岁", "风吹云乱天垂泪"]
    results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
    for result in results:
        print(result)
        
    # ['春满乾坤喜满门', '竹报平安梅报春', '春满乾坤福满门', '春满乾坤酒满樽', '春满乾坤喜满家']
    # ['雨打花残地痛心', '雨打花残地皱眉', '雨打花残地动容', '雨打霜欺地动容', '雨打花残地洒愁']
    ```

- ### 3、API

  - ```python
    def generate(texts, use_gpu=False, beam_width=5):
    ```

    - 预测API，由上联生成下联。
    - **参数**
      - texts (list[str]): 上联文本；
      - use_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**；
      - beam_width: beam search宽度，决定每个上联输出的下联数量。
    - **返回**
      - results (list[list]\[str]): 下联文本，每个上联会生成beam_width个下联。


## 四、服务部署

- PaddleHub Serving可以部署一个在线对联生成服务，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m ernie_gen_couplet -p 8866
    ```

  - 这样就完成了一个服务化API的部署，默认端口号为8866。
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    
    # 发送HTTP请求
    
    data = {'texts':["人增福寿年增岁", "风吹云乱天垂泪"],
            'use_gpu':False, 'beam_width':5}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/ernie_gen_couplet"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    # 保存结果
    results = r.json()["results"]
    for result in results:
        print(result)
    
    # ['春满乾坤喜满门', '竹报平安梅报春', '春满乾坤福满门', '春满乾坤酒满樽', '春满乾坤喜满家']
    # ['雨打花残地痛心', '雨打花残地皱眉', '雨打花残地动容', '雨打霜欺地动容', '雨打花残地洒愁']
    ```

  - 关于PaddleHub Serving更多信息参考[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  修复windows中的编码问题

* 1.0.2

  完善API的输入文本检查
  
* 1.1.0

  修复兼容性问题

  * ```shell
    $ hub install ernie_gen_couplet==1.1.0
    ```
