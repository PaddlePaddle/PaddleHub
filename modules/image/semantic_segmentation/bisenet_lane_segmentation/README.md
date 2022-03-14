# bisenet_lane_segmentation

|模型名称|bisenet_lane_segmentation|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|bisenet|
|数据集|TuSimple|
|是否支持Fine-tuning|否|
|模型大小|9.7MB|
|指标|ACC96.09%|
|最新更新日期|2021-12-03|


## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/146115316-e9ed4220-8470-432f-b3f1-549d2bcdc845.jpg" /> 
    <img src="https://user-images.githubusercontent.com/35907364/146115396-a7d19290-6117-4831-bc35-4b14ae8f90bc.png" /> 
    </p>

- ### 模型介绍

  - 车道线分割是自动驾驶算法的一个范畴，可以用来辅助进行车辆定位和进行决策，早期已有基于传统图像处理的车道线检测方法，但是随着技术的演进，车道线检测任务所应对的场景越来越多样化，目前更多的方式是寻求在语义上对车道线存在位置的检测。bisenet_lane_segmentation是一个轻量化车道线分割模型。

  - 更多详情请参考：[bisenet_lane_segmentation](https://github.com/PaddlePaddle/PaddleSeg)
  

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.2.0

    - paddlehub >= 2.1.0

    - paddleseg >= 2.3.0
    
    - Python >= 3.7+


- ### 2、安装

    - ```shell
      $ hub install bisenet_lane_segmentation
      ```
      
    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

    
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run bisenet_lane_segmentation --input_path "/PATH/TO/IMAGE"
    ```
    
  - 通过命令行方式实现hub模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

    - ```python
      import paddlehub as hub
      import cv2

      model = hub.Module(name="bisenet_lane_segmentation")
      result = model.predict(image_list=["/PATH/TO/IMAGE"])
      print(result)
      ```
- ### 3、API

    - ```python
        def predict(self, 
                    image_list, 
                    visualization, 
                    save_path):
      ```

        - 车道线分割预测API，用于将输入图片中的车道线分割出来。

        - 参数

            - image_list (list(str | numpy.ndarray)):图片输入路径或者BGR格式numpy数据。
            - visualization (bool): 是否进行可视化，默认为False。
            - save_path (str): 当visualization为True时，保存图片的路径，默认为"bisenet_lane_segmentation_output"。

        - 返回

            - result (list(numpy.ndarray))：模型分割结果：

 
## 四、服务部署

- PaddleHub Serving可以部署车道线分割在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    - ```shell
      $ hub serving start -m bisenet_lane_segmentation
      ```

    - 这样就完成了一个车道线分割在线服务API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

    ```python
    import requests
    import json
    import cv2
    import base64

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
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/bisenet_lane_segmentation"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    #print(r.json())
    mask = base64_to_cv2(r.json()["results"]['data'][0])
    print(mask)
    ```

## 五、更新历史

* 1.0.0

  初始发布

