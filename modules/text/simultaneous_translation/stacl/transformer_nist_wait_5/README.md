# transformer_nist_wait_5
|模型名称|transformer_nist_wait_5|
| :--- | :---: | 
|类别|同声传译|
|网络|transformer|
|数据集|NIST 2008-中英翻译数据集|
|是否支持Fine-tuning|否|
|模型大小|377MB|
|最新更新日期|2021-09-17|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - 同声传译（Simultaneous Translation），即在句子完成之前进行翻译，同声传译的目标是实现同声传译的自动化，它可以与源语言同时翻译，延迟时间只有几秒钟。
    STACL 是论文 [STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework](https://www.aclweb.org/anthology/P19-1289/) 中针对同传提出的适用于所有同传场景的翻译架构。
    - STACL 主要具有以下优势：

    - Prefix-to-Prefix架构拥有预测能力，即在未看到源词的情况下仍然可以翻译出对应的目标词，克服了SOV→SVO等词序差异
    <p align="center">
    <img src="https://user-images.githubusercontent.com/40840292/133761990-13e55d0f-5c3a-476c-8865-5808d13cba97.png"> <br />
    </p>
     和传统的机器翻译模型主要的区别在于翻译时是否需要利用全句的源句。上图中，Seq2Seq模型需要等到全句的源句（1-5）全部输入Encoder后，Decoder才开始解码进行翻译；而STACL架构采用了Wait-k（图中Wait-2）的策略，当源句只有两个词（1和2）输入到Encoder后，Decoder即可开始解码预测目标句的第一个词。

    - Wait-k策略可以不需要全句的源句，直接预测目标句，可以实现任意的字级延迟，同时保持较高的翻译质量。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/40840292/133762098-6ea6f3ca-0d70-4a0a-981d-0fcc6f3cd96b.png"> <br />
    </p>
     Wait-k策略首先等待源句单词，然后与源句的其余部分同时翻译，即输出总是隐藏在输入后面。这是受到同声传译人员的启发，同声传译人员通常会在几秒钟内开始翻译演讲者的演讲，在演讲者结束几秒钟后完成。例如，如果k=2，第一个目标词使用前2个源词预测，第二个目标词使用前3个源词预测，以此类推。上图中，(a)simultaneous: our wait-2 等到"布什"和"总统"输入后就开始解码预测"pres."，而(b) non-simultaneous baseline 为传统的翻译模型，需要等到整句"布什 总统 在 莫斯科 与 普京 会晤"才开始解码预测。
    
  - 该PaddleHub Module基于transformer网络结构，采用wait-5策略进行中文到英文的翻译。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.1.0

  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install transformer_nist_wait_5
    ```

  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub

    model = hub.Module(name="transformer_nist_wait_5")

    # 待预测数据（模拟同声传译实时输入）
    text = [
        "他", 
        "他还", 
        "他还说", 
        "他还说现在", 
        "他还说现在正在",
        "他还说现在正在为",
        "他还说现在正在为这",
        "他还说现在正在为这一",
        "他还说现在正在为这一会议",
        "他还说现在正在为这一会议作出",
        "他还说现在正在为这一会议作出安排",
        "他还说现在正在为这一会议作出安排。",      
    ]

    for t in text:
        print("input: {}".format(t))
        result = model.translate(t)
        print("model output: {}\n".format(result))

    # input: 他
    # model output: 
    #
    # input: 他还
    # model output: 
    #
    # input: 他还说
    # model output:
    #
    # input: 他还说现在
    # model output:
    #
    # input: 他还说现在正在
    # model output: he
    #
    # input: 他还说现在正在为
    # model output: he also
    #
    # input: 他还说现在正在为这
    # model output: he also said
    #
    # input: 他还说现在正在为这一
    # model output: he also said that
    #
    # input: 他还说现在正在为这一会议
    # model output: he also said that he
    #
    # input: 他还说现在正在为这一会议作出
    # model output: he also said that he was
    #
    # input: 他还说现在正在为这一会议作出安排
    # model output: he also said that he was making
    #
    # input: 他还说现在正在为这一会议作出安排。
    # model output: he also said that he was making arrangements for this meeting . 
    ```

- ### 2、 API

    - ```python
      __init__(max_length=256, max_out_len=256)
      ```

        - 初始化module， 可配置模型的输入文本的最大长度

        - **参数**

            - max_length(int): 输入文本的最大长度，默认值为256。
            - max_out_len(int): 输出文本的最大解码长度，超过最大解码长度时会截断句子的后半部分，默认值为256。

    - ```python
      translate(text, use_gpu=False)
      ```

        - 预测API，输入源语言的文本（模拟同传语音输入），解码后输出翻译后的目标语言文本。

        - **参数**

            - text(str): 输入源语言的文本，数据类型为str
            - use_gpu(bool): 是否使用gpu进行预测，默认为False

        - **返回**

            - result(str): 翻译后的目标语言文本。

## 四、服务部署

- PaddleHub Serving可以部署一个在线同声传译服务(需要用户配置一个语音转文本应用预先将语音输入转为中文文字)，可以将此接口用于在线web应用。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m transformer_nist_wait_5
    ```

  - 启动时会显示加载模型过程，启动成功后显示

  - ```shell
    Loading transformer_nist_wait_5 successful.
    ```

  - 这样就完成了服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果       

  - ```python
    import requests
    import json

    # 待预测数据（模拟同声传译实时输入）
    text = [
        "他", 
        "他还", 
        "他还说", 
        "他还说现在", 
        "他还说现在正在",
        "他还说现在正在为",
        "他还说现在正在为这",
        "他还说现在正在为这一",
        "他还说现在正在为这一会议",
        "他还说现在正在为这一会议作出",
        "他还说现在正在为这一会议作出安排",
        "他还说现在正在为这一会议作出安排。",      
    ]

    # 指定预测方法为transformer_nist_wait_5并发送post请求，content-type类型应指定json方式
    # HOST_IP为服务器IP
    url = "http://HOST_IP:8866/predict/transformer_nist_wait_5"
    headers = {"Content-Type": "application/json"}
    for t in text:
        print("input: {}".format(t))
        r = requests.post(url=url, headers=headers, data=json.dumps({"text": t}))
        # 打印预测结果
        print("model output: {}\n".format(result.json()['results']))

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0
    初始发布
    ```shell
    hub install transformer_nist_wait_5==1.0.0
    ```
