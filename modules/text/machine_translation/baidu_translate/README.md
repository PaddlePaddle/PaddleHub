# baidu_translate
|模型名称|baidu_translate|
| :--- | :---: |
|类别|文本-机器翻译|
|网络|-|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-09-01|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - 本模块提供百度翻译开放平台的服务，可支持多语种互译。您只需要通过传入待翻译的内容，并指定要翻译的源语言（支持源语言语种自动检测）和目标语言种类，就可以得到相应的翻译结果。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0

  - paddlehub >= 2.3.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install baidu_translate
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name='baidu_translate')
    result = module.translate("I like panda")
    print(result)
    ```

- ### 2、API

  - ```python
    def translate(query: str,
            from_lang: Optional[str] = "en",
            to_lang: Optional[int] = "zh")
    ```

    - 翻译API，输入源语言的文本句子，解码后输出翻译后的目标语言的文本句子。

    - **参数**

      - `query`(str): 待翻译的语言。
      - `from_lang`(int): 源语言。
      - `to_lang`(int): 目标语言。

    - **返回**

      - `result`(str): 翻译后的目标语言句子。

  源语言和目标语言都采用ISO 639-1语言编码标准来表示，常用的语言编码如下, 更多语言表示可以参考[文档](https://fanyi-api.baidu.com/doc/21)。
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/188076725-c2ac6831-1f9d-416a-bf9a-8f3671d6de36.png"  width = "80%" hspace='10'/>

## 四、服务部署

- 通过启动PaddleHub Serving，可以加载模型部署在线翻译服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m baidu_translate
    ```

  - 通过以上命令可完成一个翻译API的部署，默认端口号为8866。


- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    text = "I like panda"
    data = {"query": text, "from_lang":'en', "to_lang":'zh'}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/baidu_translate"
    # 指定post请求的headers为application/json方式
    headers = {"Content-Type": "application/json"}

    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    print(r.json())
    ```

  - 关于PaddleHub Serving更多信息参考：[服务部署](../../../../docs/docs_ch/tutorial/serving.md)

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install baidu_translate==1.0.0
    ```
