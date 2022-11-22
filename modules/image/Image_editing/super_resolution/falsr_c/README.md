# falsr_c


|模型名称|falsr_c|
| :--- | :---: | 
|类别|图像-图像编辑|
|网络|falsr_c|
|数据集|DIV2k|
|是否支持Fine-tuning|否|
|模型大小|4.4MB|
|PSNR|37.66|
|最新更新日期|2021-02-26|


## 一、模型基本信息

- ### 应用效果展示
  
  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/133558583-0b7049db-ed1f-4a16-8676-f2141fcb3dee.png" width = "450" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130899031-a6f8c58a-5cb7-4105-b990-8cca5ae15368.png" width = "450" height = "300" hspace='10'/>
    </p>


- ### 模型介绍

  - falsr_c是基于Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search设计的轻量化超分辨模型。该模型使用多目标方法处理超分问题，同时使用基于混合控制器的弹性搜索策略来提升模型性能。该模型提供的超分倍数为2倍。

  - 更多详情请参考：[falsr_c](https://github.com/xiaomi-automl/FALSR)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0


- ### 2、安装
    - ```shell
      $ hub install falsr_c
      ```

    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
    | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
- ### 1、命令行预测

  - ```
    $ hub run falsr_c --input_path "/PATH/TO/IMAGE"
    ```
- ### 2、预测代码示例

  ```python
  import cv2
  import paddlehub as hub

  sr_model = hub.Module(name='falsr_c')
  im = cv2.imread('/PATH/TO/IMAGE').astype('float32')
  #visualization=True可以用于查看超分图片效果，可设置为False提升运行速度。
  res = sr_model.reconstruct(images=[im], visualization=True)
  print(res[0]['data'])
  ```

- ### 3、API

  - ```python
    def reconstruct(images=None,
                    paths=None,
                    use_gpu=False,
                    visualization=False,
                    output_dir="falsr_c_output")
    ```

    - 预测API，用于图像超分辨率。

    - **参数**

      * images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      * paths (list\[str\]): 图片的路径；
      * use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 图片的保存路径。

    - **返回**

      * res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'save\_path', 'data'，对应的取值为：
        * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
        * data (numpy.ndarray): 超分辨后图像。

  - ```python
    def save_inference_model(dirname)
    ```

    - 将模型保存到指定路径。

    - **参数**

      * dirname: 模型保存路径



## 四、服务部署

- PaddleHub Serving可以部署一个图像超分的在线服务。

- ### 第一步：启动PaddleHub Serving

    - 运行启动命令：

      - ```shell
        $ hub serving start -m falsr_c
        ```

      - 这样就完成了一个超分任务的服务化API的部署，默认端口号为8866。

      - **NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

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
        data = {'images':[cv2_to_base64(org_im)]}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/falsr_c"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        sr = base64_to_cv2(r.json()["results"][0]['data'])
        cv2.imwrite('falsr_c_X2.png', sr)
        print("save image as falsr_c_X2.png")
        ```


## 五、更新历史


* 1.0.0

  初始发布


* 1.1.0

  移除 fluid API

  ```shell
  $ hub install falsr_c == 1.1.0
  ```
