# ernie_vilg

|模型名称|ernie_vilg|
| :--- | :---: |
|类别|图像-文图生成|
|网络|ERNIE-ViLG|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-08-02|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

  - 输入文本 "宁静的小镇"  风格 "油画"

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/183041589-57debf50-80ec-496f-8bb5-42d9d38646dd.png"  width = "80%" hspace='10'/>
  <br />


### 模型介绍

文心ERNIE-ViLG参数规模达到100亿，是目前为止全球最大规模中文跨模态生成模型，在文本生成图像、图像描述等跨模态生成任务上效果全球领先，在图文生成领域MS-COCO、COCO-CN、AIC-ICC等数据集上取得最好效果。你可以输入一段文本描述以及生成风格，模型就会根据输入的内容自动创作出符合要求的图像。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ernie_vilg
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run ernie_vilg --text_prompts "宁静的小镇" --output_dir ernie_vilg_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="ernie_vilg")
    text_prompts = ["宁静的小镇"]
    images = module.generate_image(text_prompts=text_prompts, output_dir='./ernie_vilg_out/')  
    ```

- ### 3、API

  - ```python
    def __init__(ak: Optional[str]=None, sk: Optional[str]=None)
    ```
    - 初始化模块，可自定义用于申请访问文心API的ak和sk。

    - **参数**
      - ak:(Optional[str]): 用于申请文心api使用token的ak，可不填。
      - sk:(Optional[str]): 用于申请文心api使用token的sk，可不填。

  - ```python
    def generate_image(
              text_prompts:str,
              style: Optional[str] = "油画",
              topk: Optional[int] = 10,
              output_dir: Optional[str] = 'ernievilg_output')
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。
      - style(Optional[str]): 生成图像的风格，当前支持'油画','水彩','粉笔画','卡通','儿童画','蜡笔画'。
      - topk(Optional[int]): 保存前多少张图，最多保存10张。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"ernievilg_output"。


    - **返回**
      - images(List(PIL.Image)): 返回生成的所有图像列表，PIL的Image格式。

## 四、服务部署

- PaddleHub Serving可以部署一个在线文图生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m ernie_vilg
    ```

  - 这样就完成了一个文图生成的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果。

  - ```python
    import requests
    import json
    import cv2
    import base64
    from io import BytesIO
    from PIL import Image

    # 发送HTTP请求
    data = {'text_prompts': '巨大的白色城堡'}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/ernie_vilg"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 获取返回结果
    for i, result in enumerate(r.json()["results"]):
      image = Image.open(BytesIO(base64.b64decode(result)))
      image.save('result_{}.png'.format(i))


## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install ernie_vilg == 1.0.0
  ```
