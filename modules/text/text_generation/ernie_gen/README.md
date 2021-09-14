# ernie_gen

| 模型名称            |   ernie_gen   |
| :------------------ | :-----------: |
| 类别                | 文本-文本生成 |
| 网络                |   ERNIE-GEN   |
| 数据集              |       -       |
| 是否支持Fine-tuning |      是       |
| 模型大小            |      85K      |
| 最新更新日期        |  2021-07-20   |
| 数据指标            |       -       |


## 一、模型基本信息

- ### 模型介绍
  - ERNIE-GEN 是面向生成任务的预训练-微调框架，首次在预训练阶段加入span-by-span 生成任务，让模型每次能够生成一个语义完整的片段。在预训练和微调中通过填充式生成机制和噪声感知机制来缓解曝光偏差问题。此外, ERNIE-GEN 采样多片段-多粒度目标文本采样策略, 增强源文本和目标文本的关联性，加强了编码器和解码器的交互。
  - ernie_gen module是一个具备微调功能的module，可以快速完成特定场景module的制作。

<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133191670-8eb1c542-f8e8-4715-adb2-6346b976fab1.png"  width="600" hspace='10'/>
</p>

- 更多详情请查看：[ERNIE-GEN:An Enhanced Multi-Flow Pre-training and Fine-tuning Framework for Natural Language Generation](https://arxiv.org/abs/2001.11314)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

  - paddlenlp >= 2.0.0					                                

- ### 2、安装

  - ```shell
    $ hub install ernie_gen
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ernie_gen需要**先针对特定数据集fine-tune**，才能使用
  - 文本生成任务有很多种，ernie_gen仅提供了生成文本的基本参数，需要针对特定任务的数据集做fine-tune才能使用
  - paddlehub提供了简单的fine-tune数据集：[train.txt](./test_data/train.txt), [dev.txt](./test_data/dev.txt)
  - paddlehub也提供了多个fine-tune好的预训练模型：[对联生成](../ernie_gen_couplet/)，[情话生成](../ernie_gen_lover_words/)，[诗歌生成](../ernie_gen_poetry/)等

### 1、Fine-tune并封装

- #### Fine-tune代码实例

  - ```python
    import paddlehub as hub
    
    module = hub.Module(name="ernie_gen")
    
    result = module.finetune(
        train_path='train.txt',
        dev_path='dev.txt',
        max_steps=300,
        batch_size=2
    )
    
    module.export(params_path=result['last_save_path'], module_name="ernie_gen_test", author="test")
    ```

- #### API说明

  - ```python
    def finetune(train_path,
                 dev_path=None,
                 save_dir="ernie_gen_result",
                 init_ckpt_path=None,
                 use_gpu=True,
                 max_steps=500,
                 batch_size=8,
                 max_encode_len=15,
                 max_decode_len=15,
                 learning_rate=5e-5,
                 warmup_proportion=0.1,
                 weight_decay=0.1,
                 noise_prob=0,
                 label_smooth=0,
                 beam_width=5,
                 length_penalty=1.0,
                 log_interval=100,
                 save_interval=200):
    ```
    
    - 微调模型参数的API
    - **参数**
      - train_path(str): 训练集路径。训练集的格式应为："序号\t输入文本\t标签"，例如："1\t床前明月光\t疑是地上霜"，注意\t不能使用空格替代
      - dev_path(str): 验证集路径。验证集的格式应为："序号\t输入文本\t标签"，例如："1\t举头望明月\t低头思故乡"，注意\t不能使用空格替代
      - save_dir(str): 模型保存以及验证集预测输出路径。
      - init_ckpt_path(str): 模型初始化加载路径，可实现增量训练。
      - use_gpu(bool): 是否使用GPU。
      - max_steps(int): 最大训练步数。
      - batch_size(int): 训练时的batch大小。
      - max_encode_len(int): 最长编码长度。
      - max_decode_len(int): 最长解码长度。
      - learning_rate(float): 学习率大小。
      - warmup_proportion(float): 学习率warmup比例。
      - weight_decay(float): 权值衰减大小。
      - noise_prob(float): 噪声概率，详见ernie gen论文。
      - label_smooth(float): 标签平滑权重。
      - beam_width(int): 验证集预测时的beam大小。
      - length_penalty(float): 验证集预测时的长度惩罚权重。
      - log_interval(int): 训练时的日志打印间隔步数。
      - save_interval(int): 训练时的模型保存间隔部署。验证集将在模型保存完毕后进行预测。
    - **返回**
      - result(dict): 运行结果。包含2个键:
        - last_save_path(str): 训练结束时的模型保存路径。
        - last_ppl(float): 训练结束时的模型困惑度。
    
  - ```python
    def export(
      params_path,
      module_name,
      author,
      version="1.0.0",
      summary="",
      author_email="",
      export_path="."):
    ```
    
    - module导出API，通过此API可以一键将训练参数打包为hub module。
    - **参数**
      - params_path(str): 模型参数路径。
      - module_name(str): module名称，例如"ernie_gen_couplet"。
      - author(str): 作者名称。
      - max_encode_len(int): 最大编码长度。
      - max_decode_len(int): 最大解码长度
      - version(str): 版本号。
      - summary(str): module的英文简介。
      - author_email(str): 作者的邮箱地址。
      - export_path(str): module的导出路径。

### 2、模型预测

- **定义`$module_name`为export指定的module_name**

- 模型转换完毕之后，通过`hub install $module_name`安装该模型，即可通过以下2种方式调用自制module：

- #### 法1：命令行预测

  - ```python
    $ hub run $module_name --input_text="输入文本" --use_gpu True --beam_width 5
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- #### 法2：API预测

  - ```python
    import paddlehub as hub
    
    module = hub.Module(name="$module_name")
    
    test_texts = ["输入文本1", "输入文本2"]
    # generate包含3个参数，texts为输入文本列表，use_gpu指定是否使用gpu，beam_width指定beam search宽度。
    results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
    for result in results:
        print(result)
    ```

- 您也可以将`$module_name`文件夹打包为tar.gz压缩包并联系PaddleHub工作人员上传至PaddleHub模型仓库，这样更多的用户可以通过一键安装的方式使用您的模型。PaddleHub非常欢迎您的贡献，共同推动开源社区成长。

## 四、服务部署

- PaddleHub Serving 可以部署一个文本生成的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m $module_name -p 8866
    ```

  - 这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 客户端通过以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    
    # 发送HTTP请求
    
    data = {'texts':["输入文本1", "输入文本2"],
            'use_gpu':True, 'beam_width':5}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/$module_name"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    
    # 保存结果
    results = r.json()["results"]
    for result in results:
        print(result)
    ```
  
- **NOTE:** 上述`$module_name`为export指定的module_name

## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  修复模型导出bug

* 1.0.2

   修复windows运行中的bug

* 1.1.0

   接入PaddleNLP
   
   - ```shell
     $ hub install ernie_gen==1.1.0
     ```
