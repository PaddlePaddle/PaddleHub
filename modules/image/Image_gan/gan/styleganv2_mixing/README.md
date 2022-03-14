# styleganv2_mixing

|模型名称|styleganv2_mixing|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|StyleGAN V2|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|190MB|
|最新更新日期|2021-12-23|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/147241001-3babb1bd-98d4-4a9c-a61d-2298fca041e1.jpg"  width = "40%"  hspace='10'/>
    <br />
    输入图像1
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/147241006-0bc2cda8-d271-4cfd-8a0d-e6feea7bf167.jpg"  width = "40%"  hspace='10'/>
    <br />
    输入图像2
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/147241020-f4420729-c489-4661-b43f-c929c62c0ce7.png"  width = "40%"  hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - StyleGAN V2 的任务是使用风格向量进行image generation，而Mixing模块则是利用其风格向量实现两张生成图像不同层次不同比例的混合。



## 二、安装

- ### 1、环境依赖  
  - paddlepaddle >= 2.1.0
  - paddlehub >= 2.1.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install styleganv2_mixing
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run styleganv2_mixing --image1 "/PATH/TO/IMAGE1" --image2 "/PATH/TO/IMAGE2"
    ```
  - 通过命令行方式实现人脸融合模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="styleganv2_mixing")
    input_path = ["/PATH/TO/IMAGE"]
    # Read from a file
    module.generate(paths=input_path, direction_name = 'age', direction_offset = 5, output_dir='./editing_result/', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    generate(self, images=None, paths=None, weights = [0.5] * 18, output_dir='./mixing_result/', use_gpu=False, visualization=True)
    ```
    - 人脸融合生成API。

    - **参数**
      - images (list[dict]): data of images, 每一个元素都为一个 dict，有关键字 image1, image2, 相应取值为：
          - image1 (numpy.ndarray): 待融合的图片1，shape 为 \[H, W, C\]，BGR格式；<br/>
          - image2 (numpy.ndarray) : 待融合的图片2，shape为 \[H, W, C\]，BGR格式；<br/>
      - paths (list[str]): paths to images, 每一个元素都为一个dict, 有关键字 image1, image2, 相应取值为：
          - image1 (str): 待融合的图片1的路径；<br/>
          - image2 (str) : 待融合的图片2的路径；<br/>
      - weights (list(float)): 融合的权重
      - images (list\[numpy.ndarray\]): 图片数据 <br/>
      - paths (list\[str\]): 图片路径；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹


## 四、服务部署

- PaddleHub Serving可以部署一个在线人脸融合服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m styleganv2_mixing
    ```

  - 这样就完成了一个人脸融合的在线服务API的部署，默认端口号为8866。

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
    data = {'images':[{'image1': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE1")),'image2': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE2"))}]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/styleganv2_mixing"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install styleganv2_mixing==1.0.0
    ```
