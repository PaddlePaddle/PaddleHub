# Vehicle_License_Plate_Recognition

|模型名称|Vehicle_License_Plate_Recognition|
| :--- | :---: |
|类别|图像 - 文字识别|
|网络|-|
|数据集|CCPD|
|是否支持Fine-tuning|否|
|模型大小|111MB|
|最新更新日期|2021-03-22|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/35a3dab32ac948549de41afba7b51a5770d3f872d60b437d891f359a5cef8052"  width = "450" height = "300" hspace='10'/> <br />
    </p>


- ### 模型介绍

  - Vehicle_License_Plate_Recognition是一个基于CCPD数据集训练的车牌识别模型，能够检测出图像中车牌位置并识别其中的车牌文字信息。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0  

  - paddlehub >= 2.0.4

  - paddleocr >= 2.0.2  

- ### 2、安装

  - ```shell
    $ hub install Vehicle_License_Plate_Recognition
    ```

## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="Vehicle_License_Plate_Recognition")
    result = model.plate_recognition(images=[cv2.imread('/PATH/TO/IMAGE')])
    ```

- ### 2、API

  - ```python
    def plate_recognition(images)
    ```

    - 车牌识别 API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>


    - **返回**
      - results(list(dict{'license', 'bbox'})): 识别到的车牌信息列表，包含车牌的位置坐标和车牌号码


## 四、服务部署

- PaddleHub Serving可以部署一个在线车牌识别服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m Vehicle_License_Plate_Recognition
    ```

  - 这样就完成了一个车牌识别的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/Vehicle_License_Plate_Recognition"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install Vehicle_License_Plate_Recognition==1.0.0
    ```
