# photo_restoration

|模型名称|photo_restoration|
| :--- | :---: | 
|类别|图像-图像编辑|
|网络|基于deoldify和realsr模型|
|是否支持Fine-tuning|否|
|模型大小|64MB+834MB|
|指标|-|
|最新更新日期|2021-08-19|



## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130897828-d0c86b81-63d1-4e9a-8095-bc000b8c7ca8.jpg" width = "260" height = "400" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130897762-5c9fa711-62bc-4067-8d44-f8feff8c574c.png" width = "260" height = "400" hspace='10'/>
    </p>



- ### 模型介绍

    - photo_restoration 是针对老照片修复的模型。它主要由两个部分组成：着色和超分。着色模型基于deoldify
    ，超分模型基于realsr. 用户可以根据自己的需求选择对图像进行着色或超分操作。因此在使用该模型时，请预先安装deoldify和realsr两个模型。

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

    - NOTE: 使用该模型需要自行安装ffmpeg，若您使用conda环境，推荐使用如下语句进行安装。

      ```shell
      $ conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
      ```
   
- ### 2、安装
    - ```shell
      $ hub install photo_restoration
      ```
      
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测
  - ### 1、预测代码示例

    - ```python
      import cv2
      import paddlehub as hub

      model = hub.Module(name='photo_restoration', visualization=True)
      im = cv2.imread('/PATH/TO/IMAGE')
      res = model.run_image(im)

      ```
- ### 2、API


    ```python
    def run_image(self,
                  input,
                  model_select= ['Colorization', 'SuperResolution'],
                  save_path = 'photo_restoration'):
    ```

    - 预测API，用于图片修复。

    - **参数**

        - input (numpy.ndarray｜str): 图片数据，numpy.ndarray 或者 str形式。ndarray.shape 为 \[H, W, C\]，BGR格式; str为图片的路径。

        - model_select (list\[str\]): 选择对图片对操作，\['Colorization'\]对图像只进行着色操作， \['SuperResolution'\]对图像只进行超分操作；
        默认值为\['Colorization', 'SuperResolution'\]。

        - save_path (str): 保存图片的路径, 默认为'photo_restoration'。

    - **返回**

        - output (numpy.ndarray): 照片修复结果，ndarray.shape 为 \[H, W, C\]，BGR格式。



## 四、服务部署

- PaddleHub Serving可以部署一个照片修复的在线服务。

- ## 第一步：启动PaddleHub Serving

    - 运行启动命令：

        - ```shell
          $ hub serving start -m photo_restoration
          ```

        - 这样就完成了一个照片修复的服务化API的部署，默认端口号为8866。

        - **NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

    - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

        ```python
        import requests
        import json
        import base64

        import cv2
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
        org_im = cv2.imread('PATH/TO/IMAGE')
        data = {'images':cv2_to_base64(org_im), 'model_select': ['Colorization', 'SuperResolution']}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/photo_restoration"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        img = base64_to_cv2(r.json()["results"])
        cv2.imwrite('PATH/TO/SAVE/IMAGE', img)
        ```

## 五、更新历史


* 1.0.0

  初始发布

* 1.0.1

  适配paddlehub2.0版本
