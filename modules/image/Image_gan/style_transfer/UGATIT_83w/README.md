# UGATIT_83w

|模型名称|UGATIT_83w|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|U-GAT-IT|
|数据集|selfie2anime|
|是否支持Fine-tuning|否|
|模型大小|41MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/136651638-33cac040-edad-41ac-a9ce-7c0e678d8c52.jpg" width = "400" height = "400" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/136651644-dd1d3836-99b3-40f0-8543-37de18f9cfd9.jpg" width = "400" height = "400" hspace='10'/>
    </p>



- ### 模型介绍

  - UGATIT 图像风格转换模型, 模型可将输入的人脸图像转换成动漫风格.


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.8.2  

  - paddlehub >= 1.8.0

- ### 2、安装

  - ```shell
    $ hub install UGATIT_83w
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 
 
## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub

    # 模型加载
    # use_gpu：是否使用GPU进行预测
    model = hub.Module(name='UGATIT_83w', use_gpu=False)

    # 模型预测
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(
        self,
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False
    )
    ```

    - 风格转换API，将输入的人脸图像转换成动漫风格。

    - **参数**
        * images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，默认为 None；
        * paths (list\[str\]): 图片的路径，默认为 None；
        * batch\_size (int): batch 的大小，默认设为 1；
        * visualization (bool): 是否将识别结果保存为图片文件，默认设为 False；
        * output\_dir (str): 图片的保存路径，默认设为 output

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**

      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]
      

## 四、服务部署

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  
  - ```shell
    $ hub serving start -m UGATIT_83w
    ```

  - 这样就完成了一个图像风格转换的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

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
    url = "http://127.0.0.1:8866/predict/UGATIT_83w"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布