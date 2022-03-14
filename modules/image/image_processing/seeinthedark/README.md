# seeinthedark

|模型名称|seeinthedark|
| :--- | :---: |
|类别|图像 - 暗光增强|
|网络|ConvNet|
|数据集|SID dataset|
|是否支持Fine-tuning|否|
|模型大小|120MB|
|最新更新日期|2021-11-02|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/142962370-a957d7b3-8050-4f5a-8462-3d6e49facb33.png"  width = "450" height = "300" hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/142962460-4a1b31ef-0eec-423b-ab3d-8622f3e8261a.png"  width = "450" height = "300" hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - 通过大量暗光条件下短曝光和长曝光组成的图像对，以RAW图像为输入，RGB图像为参照进行训练，该模型实现端到端直接将暗光下的RAW图像处理得到可见的RGB图像。

  - 更多详情参考：[Learning to See in the Dark](http://cchen156.github.io/paper/18CVPR_SID.pdf)



## 二、安装

- ### 1、环境依赖  
  - rawpy

- ### 2、安装

  - ```shell
    $ hub install seeinthedark
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a raw(Sony, .ARW) file
    $ hub run seeinthedark --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现暗光增强模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    denoiser = hub.Module(name="seeinthedark")
    input_path = "/PATH/TO/IMAGE"
    # Read from a raw file
    denoiser.denoising(paths=[input_path], output_path='./denoising_result.png', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    def denoising(images=None, paths=None, output_dir='./denoising_result/', use_gpu=False, visualization=True)
    ```
    - 暗光增强API，完成对暗光RAW图像的降噪并处理生成RGB图像。

    - **参数**
      - images (list\[numpy.ndarray\]): 输入的图像，单通道的马赛克图像; <br/>
      - paths (list\[str\]): 暗光图像文件的路径，Sony的RAW格式；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹

## 四、服务部署

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m seeinthedark
    ```

  - 这样就完成了一个图像风格转换的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import rawpy
    import base64


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(rawpy.imread("/PATH/TO/IMAGE").raw_image_visible)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/seeinthedark/"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install seeinthedark==1.0.0
    ```
