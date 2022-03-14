# animegan_v2_shinkai_33

|模型名称|animegan_v2_shinkai_33|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|AnimeGAN|
|数据集|Your Name, Weathering with you|
|是否支持Fine-tuning|否|
|模型大小|9.4MB|
|最新更新日期|2021-07-30|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/bd002c4bb6a7427daf26988770bb18648b7d8d2bfd6746bfb9a429db4867727f"  width = "450" height = "300" hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/776a84a0d97c452bbbe479592fbb8f5c6fe9c45f3b7e41fd8b7da80bf52ee668"  width = "450" height = "300" hspace='10'/>
    <br />
    输出图像
    <br />
    </p>


- ### 模型介绍

  - AnimeGAN V2 图像风格转换模型, 模型可将输入的图像转换成新海诚动漫风格，模型权重转换自[AnimeGAN V2官方开源项目](https://github.com/TachibanaYoshino/AnimeGANv2)。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.8.0  

  - paddlehub >= 1.8.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install animegan_v2_shinkai_33
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="animegan_v2_shinkai_33")
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       output_dir='output',
                       visualization=False,
                       min_size=32,
                       max_size=1024)
    ```

    - 风格转换API，将输入的图片转换为漫画风格。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - visualization (bool): 是否将识别结果保存为图片文件；<br/>
      - min\_size (int): 输入图片的短边最小尺寸，默认设为 32；<br/>
      - max\_size (int): 输入图片的短边最大尺寸，默认设为 1024。

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]


## 四、服务部署

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m animegan_v2_shinkai_33
    ```

  - 这样就完成了一个图像风格转换的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import cv2
    import base64


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/animegan_v2_shinkai_33"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  适配paddlehub2.0

* 1.0.2

  删除batch_size选项

  - ```shell
    $ hub install animegan_v2_shinkai_33==1.0.2
    ```
