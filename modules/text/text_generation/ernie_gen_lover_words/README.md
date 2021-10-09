# ernie_gen_lover_words

| 模型名称            | ernie_gen_lover_words |
| :------------------ | :-------------------: |
| 类别                |     文本-文本生成     |
| 网络                |       ERNIE-GEN       |
| 数据集              |  网络情诗、情话数据   |
| 是否支持Fine-tuning |          否           |
| 模型大小            |         420M          |
| 最新更新日期        |      2021-02-26       |
| 数据指标            |           -           |

## 一、模型基本信息

- ### 模型介绍
  - ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。
  - ernie_gen_lover_words采用网络搜集的情诗、情话数据微调，可用于生成情话。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133191670-8eb1c542-f8e8-4715-adb2-6346b976fab1.png"  width="600" hspace='10'/>
</p>

- 更多详情请参考：[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 1.8.2
  
  - paddlehub >= 1.7.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ernie_gen_lover_words
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run ernie_gen_lover_words --input_text "情人节" --use_gpu True --beam_width 5
    ```
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
  
- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    
    module = hub.Module(name="ernie_gen_lover_words")
    
    test_texts = ['情人节', '故乡', '小编带大家了解一下程序员情人节']
    results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
    for result in results:
        print(result)
        
    # '情人节，我愿做一条鱼，任你红烧、白煮、清蒸，然后躺在你温柔的胃里。', '情人节，对你的思念太重，压断了电话线，烧坏了手机卡，掏尽了钱包袋，吃光了安眠药，哎!可我还是思念你。', '情人节，对你的思念太重，压断了电话线，烧坏了手机卡，掏尽了钱包袋，吃光了安眠药，哎!可我还是思念你，祝你情人节快乐!', '情人节，对你的思念太重，压断了电话线，烧坏了手机卡，掏尽了钱包袋，吃光了安眠药，唉!可我还是思念你，祝你情人节快乐!', '情人节，对你的思念太重，压断了电话线，烧坏了手机卡，掏尽了钱包袋，吃光了安眠药，哎!可是我还是思念你。']
    # ['故乡，是深秋的雨，云雾缭绕，夏日的阳光照耀下，像一只只翅膀，那就是思念。', '故乡，是深秋的雨，是诗人们吟咏的乡村序曲，但愿天下有情人，一定难忘。', '故乡，是深秋的雨，是诗人们吟咏的一篇美丽的诗章，但愿天下有情人，都一定走进了蒙蒙细雨中。', '故乡，是深秋的雨，是诗人们吟咏的一篇美丽的诗章，但愿天下有情人，都一定走进了蒙蒙的细雨，纷纷而来。', '故乡，是深秋的雨，是诗人们吟咏的一篇美丽的诗章，但愿天下有情人，都一定走进了蒙蒙的细雨中。']
    # ['小编带大家了解一下程序员情人节，没有人会悄悄的下载数据库，没有人会升级!希望程序可以好好的工作!', '小编带大家了解一下程序员情人节，没有人会悄悄的下载数据库，没有人会升级!希望程序可以重新拥有!', '小编带大家了解一下程序员情人节，没有人会悄悄的下载数据库，没有人会升级!希望程序可以好好的工作。', '小编带大家了解一下程序员情人节，没有人会悄悄的下载数据库，没有人会升级!希望程序可以重新把我们送上。', '小编带大家了解一下程序员情人节，没有人会悄悄的下载数据库，没有人会升级!希望程序可以重新把我们送上!']
    ```

- ### 3、API

  - ```python
    def generate(texts, use_gpu=False, beam_width=5):
    ```
    
    - 预测API，输入情话开头，输出情话下文。

    - **参数**
      - texts(list[str]):情话的开头
      - use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
      - beam_width: beam search宽度，决定每个情话开头输出的下文数目
    
    - **返回**
      - results(list[list]\[str]): 情话下文，每个情话开头会生成beam_width个下文


## 四、服务部署

- PaddleHub Serving可以部署一个在线文本生成服务，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m ernie_gen_lover_words -p 8866  
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。
  
  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。


- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    
    # 发送HTTP请求
    
    data = {'texts':['情人节', '故乡', '小编带大家了解一下程序员情人节'],
            'use_gpu':False, 'beam_width':5}
    headers = {"Content-type": "application/json"}
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/ernie_gen_lover_words"
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

* 1.0.1

  完善API的输入文本检查
  
  - ```shell
    $ hub install ernie_gen_lover_words==1.0.1
    ```
