# realsr

|模型名称|reasr|
| :--- | :---: | 
|类别|图像-图像编辑|
|网络|LP-KPN|
|数据集|RealSR dataset|
|是否支持Fine-tuning|否|
|模型大小|64MB|
|PSNR|29.05|
|最新更新日期|2021-02-26|



## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/133558583-0b7049db-ed1f-4a16-8676-f2141fcb3dee.png" width = "450" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130789888-a0d4f78e-acd6-44c1-9570-7390e90ae8dc.png" width = "450" height = "300" hspace='10'/>
    </p>

- ### 模型介绍

  - realsr是用于图像和视频超分模型，该模型基于Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Mode，它能够将输入的图片和视频超分四倍。
  
  - 更多详情请参考：[realsr](https://github.com/csjcai/RealSR)
  

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
      $ hub install realsr
      ```
      
    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

    


## 三、模型API预测

  - ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(name='realsr')
    model.predict('/PATH/TO/IMAGE/OR/VIDEO')
    ```
  - ### 2、API

    - ```python
      def predict(self, input):
      ```

      - 超分API，得到超分后的图片或者视频。


      - **参数**

        - input (str): 图片或者视频的路径；

      - **返回**

        - 若输入是图片，返回值为：
          - pred_img(np.ndarray): BGR图片数据；
          - out_path(str): 保存图片路径。

        - 若输入是视频，返回值为：
          - frame_pattern_combined(str): 视频超分后单帧数据保存路径；
          - vid_out_path(str): 视频保存路径。

    - ```python
      def run_image(self, img):
      ```
      - 图像超分API， 得到超分后的图片。

      - **参数**

        - img (str｜np.ndarray): 图片路径或则BGR格式图片。

      - **返回**

        - pred_img(np.ndarray): BGR图片数据；

    - ```python
      def run_video(self, video):
      ```
       - 视频超分API， 得到超分后的视频。

       - **参数**

         - video(str): 待处理视频路径。

       - **返回**

         - frame_pattern_combined(str): 视频超分后单帧数据保存路径；
         - vid_out_path(str): 视频保存路径。

## 四、服务部署

- PaddleHub Serving可以部署一个在线照片超分服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    - ```shell
      $ hub serving start -m realsr
      ```

    - 这样就完成了一个图像超分的在线服务API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

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
      org_im = cv2.imread('/PATH/TO/IMAGE')
      data = {'images':cv2_to_base64(org_im)}
      headers = {"Content-type": "application/json"}
      url = "http://127.0.0.1:8866/predict/realsr"
      r = requests.post(url=url, headers=headers, data=json.dumps(data))
      img = base64_to_cv2(r.json()["results"])
      cv2.imwrite('/PATH/TO/SAVE/IMAGE', img)

      ```


## 五、更新历史

* 1.0.0

  初始发布

* 1.0.1

  适配paddlehub2.0版本

