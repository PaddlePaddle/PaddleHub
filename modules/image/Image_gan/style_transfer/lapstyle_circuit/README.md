# lapstyle_circuit

|模型名称|lapstyle_circuit|
| :--- | :---: |
|类别|图像 - 风格迁移|
|网络|LapStyle|
|数据集|COCO|
|是否支持Fine-tuning|否|
|模型大小|121MB|
|最新更新日期|2021-12-07|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/144995283-77ddba45-9efe-4f72-914c-1bff734372ed.png"  width = "50%"  hspace='10'/>
    <br />
    输入内容图形
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/144997574-8b4028ad-d871-4caf-87d1-191582bba805.jpg"  width = "50%" hspace='10'/>
    <br />
    输入风格图形
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/144997589-407a12b9-95bf-44e7-b558-b1026ef3cd5a.png"  width = "50%"  hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - LapStyle--拉普拉斯金字塔风格化网络，是一种能够生成高质量风格化图的快速前馈风格化网络，能渐进地生成复杂的纹理迁移效果，同时能够在512分辨率下达到100fps的速度。可实现多种不同艺术风格的快速迁移，在艺术图像生成、滤镜等领域有广泛的应用。

  - 更多详情参考：[Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer](https://arxiv.org/pdf/2104.05376.pdf)



## 二、安装

- ### 1、环境依赖  
  - ppgan

- ### 2、安装

  - ```shell
    $ hub install lapstyle_circuit
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run lapstyle_circuit --content "/PATH/TO/IMAGE" --style "/PATH/TO/IMAGE1"
    ```
  - 通过命令行方式实现风格转换模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="lapstyle_circuit")
    content = cv2.imread("/PATH/TO/IMAGE")
    style = cv2.imread("/PATH/TO/IMAGE1")
    results = module.style_transfer(images=[{'content':content, 'style':style}], output_dir='./transfer_result', use_gpu=True)
    ```

- ### 3、API

  - ```python
    style_transfer(images=None, paths=None, output_dir='./transfer_result/', use_gpu=False, visualization=True)
    ```
    - 风格转换API。

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

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m lapstyle_circuit
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
    data = {'images':[{'content': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE")), 'style': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE1"))}]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/lapstyle_circuit"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install lapstyle_circuit==1.0.0
    ```
