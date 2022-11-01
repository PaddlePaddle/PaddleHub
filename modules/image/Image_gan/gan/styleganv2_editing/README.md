# styleganv2_editing

|模型名称|styleganv2_editing|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|StyleGAN V2|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|190MB|
|最新更新日期|2021-12-15|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/146483720-fb0ea3c0-b259-4ad6-b176-966675b9b164.png"  width = "40%"  hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/146483730-3104795e-4ee6-43de-b4dc-b7760d502b50.png"  width = "40%"  hspace='10'/>
    <br />
    输出图像(修改age)
     <br />
    </p>

- ### 模型介绍

  - StyleGAN V2 的任务是使用风格向量进行image generation，而Editing模块则是利用预先对多图的风格向量进行分类回归得到的属性操纵向量来操纵生成图像的属性。



## 二、安装

- ### 1、环境依赖  
  - ppgan

- ### 2、安装

  - ```shell
    $ hub install styleganv2_editing
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run styleganv2_editing --input_path "/PATH/TO/IMAGE" --direction_name age --direction_offset 5
    ```
  - 通过命令行方式实现人脸编辑模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="styleganv2_editing")
    input_path = ["/PATH/TO/IMAGE"]
    # Read from a file
    module.generate(paths=input_path, direction_name='age', direction_offset=5, output_dir='./editing_result/', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    generate(self, images=None, paths=None, direction_name='age', direction_offset=0.0, output_dir='./editing_result/', use_gpu=False, visualization=True)
    ```
    - 人脸编辑生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据 <br/>
      - paths (list\[str\]): 图片路径；<br/>
      - direction_name （str): 要编辑的属性名称，对于ffhq-conf-f有预先准备的这些属性: age、eyes_open、eye_distance、eye_eyebrow_distance、eye_ratio、gender、lip_ratio、mouth_open、mouth_ratio、nose_mouth_distance、nose_ratio、nose_tip、pitch、roll、smile、yaw <br/>
      - direction_offset (float): 属性的偏移强度 <br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹


## 四、服务部署

- PaddleHub Serving可以部署一个在线人脸编辑服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m styleganv2_editing
    ```

  - 这样就完成了一个人脸编辑的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/styleganv2_editing"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install styleganv2_editing==1.0.0
    ```
