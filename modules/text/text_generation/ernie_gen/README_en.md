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

- ernie_gen can be used **only if it is first targeted at the specific dataset fine-tune**
  - There are many types of text generation tasks, ernie_gen only provides the basic parameters for text generation, which can only be used after fine-tuning the dataset for a specific task
  - Paddlehub provides a simple fine-tune dataset:[train.txt](./test_data/train.txt), [dev.txt](./test_data/dev.txt)
  - Paddlehub also offers multiple fine-tune pre-training models that work well:[Couplet generated](../ernie_gen_couplet/)，[Lover words generated](../ernie_gen_lover_words/)，[Poetry generated](../ernie_gen_poetry/)等

### 1、Fine-tune and encapsulation

- #### Fine-tune Code Example

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

- #### API Instruction

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
    
    - Fine tuning model parameters API
    - **Parameter**
      - train_path(str): Training set path. The format of the training set should be: "serial number\tinput text\tlabel", such as "1\t床前明月光\t疑是地上霜", note that \t cannot be replaced by Spaces
      - dev_path(str): validation set path. The format of the validation set should be: "serial number\tinput text\tlabel, such as "1\t举头望明月\t低头思故乡", note that \t cannot be replaced by Spaces
      - save_dir(str): Model saving and validation sets predict output paths.
      - init_ckpt_path(str): The model initializes the loading path to realize incremental training.
      - use_gpu(bool): use gpu or not
      - max_steps(int): Maximum training steps.
      - batch_size(int): Batch size during training.
      - max_encode_len(int): Maximum encoding length.
      - max_decode_len(int): Maximum decoding length.
      - learning_rate(float): Learning rate size.
      - warmup_proportion(float): Warmup rate.
      - weight_decay(float): Weight decay size.
      - noise_prob(float): Noise probability, refer to the Ernie Gen's paper.
      - label_smooth(float): Label smoothing weight.
      - beam_width(int): Beam size of validation set at the time of prediction.
      - length_penalty(float): Length penalty weight for validation set prediction.
      - log_interval(int): Number of steps at a training log printing interval.
      - save_interval(int): training model save interval deployment. The validation set will make predictions after the model is saved.
    - **Return**
      - result(dict): Run result. Contains 2 keys:
        - last_save_path(str): Save path of model at the end of training.
        - last_ppl(float): Model confusion at the end of training.
    
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
    
    - Module exports an API through which training parameters can be packaged into a Hub Module with one click.
    - **Parameter**
      - params_path(str): Module parameter path.
      - module_name(str): module name, such as "ernie_gen_couplet"。
      - author(str): Author name
      - max_encode_len(int): Maximum encoding length.
      - max_decode_len(int): Maximum decoding length.
      - version(str): The version number.
      - summary(str): English introduction to Module.
      - author_email(str): Email address of the author.
      - export_path(str): Module export path.

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
