# baidu_language_recognition
|模型名称|baidu_language_recognition|
| :--- | :---: |
|类别|文本-语种识别|
|网络|-|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-09-01|
|数据指标|-|

## 一、模型基本信息

- ### 模型介绍

  - 本模块提供百度翻译开放平台的服务，可支持语种识别。您只需要通过传入文本内容，就可以得到识别出来的语种类别。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.1.0

  - paddlehub >= 2.3.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install baidu_language_recognition
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name='baidu_language_recognition')
    result = module.recognize("I like panda")
    print(result)
    ```

- ### 2、API

  - ```python
    def recognize(query: str)
    ```

    - 语种识别API，输入文本句子，输出识别后的语种编码。

    - **参数**

      - `query`(str): 待识别的语言。

    - **返回**

      - `result`(str): 识别的结果，语言的ISO 639-1编码。

  目前支持识别的语种如下：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/188105543-21610399-23de-471b-ab60-82c3e95660a6.png"  width = "80%" hspace='10'/>

## 四、服务部署

- 通过启动PaddleHub Serving，可以加载模型部署在线语种识别服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

  - ```shell
    $ hub serving start -m baidu_language_recognition
    ```

  - 通过以上命令可完成一个语种识别API的部署，默认端口号为8866。


- ## 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json

    text = "I like panda"
    data = {"query": text}
    # 发送post请求，content-type类型应指定json方式，url中的ip地址需改为对应机器的ip
    url = "http://127.0.0.1:8866/predict/baidu_language_recognition"
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
    $ hub install baidu_language_recognition==1.0.0
    ```
