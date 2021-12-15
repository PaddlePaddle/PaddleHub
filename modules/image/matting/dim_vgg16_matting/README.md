# dim_vgg16_matting

|模型名称|dim_vgg16_matting|
| :--- | :---: | 
|类别|图像-抠图|
|网络|dim_vgg16|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|164MB|
|指标|SAD112.73|
|最新更新日期|2021-12-03|


## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/144574288-28671577-8d5d-4b20-adb9-fe737015c841.jpg" width = "337" height = "505" hspace='10' /> 
    <img src="https://user-images.githubusercontent.com/35907364/144779164-47146d3a-58c9-4a38-b968-3530aa9a0137.png" width = "337" height = "505" hspace='10'/> 
    </p>

- ### 模型介绍

  - Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。dim_vgg16_matting是一种需要trimap作为输入的matting模型。


  
  - 更多详情请参考：[dim_vgg16_matting](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3/contrib/Matting)
  

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.2.0

    - paddlehub >= 2.1.0

    - paddleseg >= 2.3.0


- ### 2、安装

    - ```shell
      $ hub install dim_vgg16_matting
      ```
      
    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

    
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run dim_vgg16_matting --input_path "/PATH/TO/IMAGE" --trimap_path "/PATH/TO/TRIMAP"
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

    - ```python
      import paddlehub as hub
      import cv2

      model = hub.Module(name="dim_vgg16_matting")

      result = model.predict(image_list=["/PATH/TO/IMAGE"], trimap_list=["PATH/TO/TRIMAP"])
      print(result)
      ```
- ### 3、API

    - ```python
        def predict(self, 
                    image_list, 
                    trimap_list, 
                    visualization, 
                    save_path):
      ```

        - 人像matting预测API，用于将输入图片中的人像分割出来。

        - 参数

            - image_list (list(str | numpy.ndarray)):图片输入路径或者BGR格式numpy数据。
            - trimap_list(list(str | numpy.ndarray)):trimap输入路径或者单通道灰度图片。
            - visualization (bool): 是否进行可视化，默认为False。
            - save_path (str): 当visualization为True时，保存图片的路径，默认为"dim_vgg16_matting_output" 。

        - 返回

            - result (list(numpy.ndarray))：模型分割结果：

 
## 四、服务部署

- PaddleHub Serving可以部署人像matting在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    - ```shell
      $ hub serving start -m dim_vgg16_matting
      ```

    - 这样就完成了一个人像matting在线服务API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json
    import cv2
    import base64
    import time
    import numpy as np

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')


    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))], 'trimaps':[cv2_to_base64(cv2.imread("/PATH/TO/TRIMAP"))]}

    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/dim_vgg16_matting"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    for image in r.json()["results"]['data']:
        data = base64_to_cv2(image)
        image_path =str(time.time()) + ".png"
        cv2.imwrite(image_path, data)
      ```

## 五、更新历史

* 1.0.0

  初始发布
