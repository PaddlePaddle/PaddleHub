# psgan

|模型名称|psgan|
| :--- | :---: |
|类别|图像 - 妆容迁移|
|网络|PSGAN|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|121MB|
|最新更新日期|2021-12-07|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/157190651-595b6964-97c5-4b0b-ac0a-c30c8520a972.png"  width = "30%"  hspace='10'/>
    <br />
    输入内容图形
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/145003966-c5c2e6ad-d306-4eaf-89a2-965a3dbf3675.jpg"  width = "30%" hspace='10'/>
    <br />
    输入妆容图形
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/157190800-b1dd79d4-0eca-4b36-b091-6fcd2c00dcf6.png"  width = "30%"  hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - PSGAN模型的任务是妆容迁移， 即将任意参照图像上的妆容迁移到不带妆容的源图像上。很多人像美化应用都需要这种技术。

  - 更多详情参考：[PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer](https://arxiv.org/pdf/1909.06956.pdf)



## 二、安装

- ### 1、环境依赖  
  - ppgan
  - dlib

- ### 2、安装

  - ```shell
    $ hub install psgan
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run psgan --content "/PATH/TO/IMAGE" --style "/PATH/TO/IMAGE1"
    ```
  - 通过命令行方式实现妆容转换模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="psgan")
    content = cv2.imread("/PATH/TO/IMAGE")
    style = cv2.imread("/PATH/TO/IMAGE1")
    results = module.makeup_transfer(images=[{'content':content, 'style':style}], output_dir='./transfer_result', use_gpu=True)
    ```

- ### 3、API

  - ```python
    makeup_transfer(images=None, paths=None, output_dir='./transfer_result/', use_gpu=False, visualization=True)
    ```
    - 妆容风格转换API。

    - **参数**

      - images (list[dict]): data of images, 每一个元素都为一个 dict，有关键字 content, style, 相应取值为：
        - content (numpy.ndarray): 待转换的图片，shape 为 \[H, W, C\]，BGR格式；<br/>
        - style (numpy.ndarray) : 风格图像，shape为 \[H, W, C\]，BGR格式；<br/>
      - paths (list[str]): paths to images, 每一个元素都为一个dict, 有关键字 content, style, 相应取值为：
        - content (str): 待转换的图片的路径；<br/>
        - style (str) : 风格图像的路径；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹


## 四、服务部署

- PaddleHub Serving可以部署一个在线妆容风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m psgan
    ```

  - 这样就完成了一个妆容风格转换的在线服务API的部署，默认端口号为8866。

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
    data = {'images':[{'content': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE")), 'style': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE1"))}]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/psgan"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install psgan==1.0.0
    ```
