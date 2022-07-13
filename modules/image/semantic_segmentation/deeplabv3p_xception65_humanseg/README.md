# deeplabv3p_xception65_humanseg

|模型名称|deeplabv3p_xception65_humanseg|
| :--- | :---: |
|类别|图像-图像分割|
|网络|deeplabv3p|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|162MB|
|指标|-|
|最新更新日期|2021-02-26|

## 一、模型基本信息

- ### 应用效果展示

  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130913092-312a5f37-842e-4fd0-8db4-5f853fd8419f.jpg" width = "337" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130913256-41056b21-1c3d-4ee2-b481-969c94754609.png" width = "337" height = "505" hspace='10'/>
    </p>

- ### 模型介绍

  - DeepLabv3+使用百度自建数据集进行训练，可用于人像分割，支持任意大小的图片输入。
<p align="center">
<img src="https://paddlehub.bj.bcebos.com/paddlehub-img/deeplabv3plus.png" hspace='10'/> <br />
</p>

- 更多详情请参考：[deeplabv3p](https://github.com/PaddlePaddle/PaddleSeg)

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install deeplabv3p_xception65_humanseg
      ```

    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1.命令行预测

  ```shell
  hub run deeplabv3p_xception65_humanseg --input_path "/PATH/TO/IMAGE"
  ```



- ### 2.预测代码示例

  ```python
  import paddlehub as hub
  import cv2

  human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
  result = human_seg.segmentation(images=[cv2.imread('/PATH/TO/IMAGE')])

  ```

- ### 3.API

    ```python
    def segmentation(images=None,
                    paths=None,
                    batch_size=1,
                    use_gpu=False,
                    visualization=False,
                    output_dir='humanseg_output')
    ```

    - 预测API，用于人像分割。

    - **参数**

      * images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      * paths (list\[str\]): 图片的路径；
      * batch\_size (int): batch 的大小；
      * use\_gpu (bool): 是否使用 GPU；
      * visualization (bool): 是否将识别结果保存为图片文件；
      * output\_dir (str): 图片的保存路径。

    - **返回**

      * res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'save\_path', 'data'，对应的取值为：
      * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
      * data (numpy.ndarray): 人像分割结果，仅包含Alpha通道，取值为0-255 (0为全透明，255为不透明)，也即取值越大的像素点越可能为人体，取值越小的像素点越可能为背景。

    ```python
    def save_inference_model(dirname,
                            model_filename=None,
                            params_filename=None,
                            combined=True)
    ```

    - 将模型保存到指定路径。

    - **参数**

      * dirname: 存在模型的目录名称
      * model\_filename: 模型文件名称，默认为\_\_model\_\_
      * params\_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
      * combined: 是否将参数保存到统一的一个文件中


## 四、服务部署

- PaddleHub Serving可以部署一个人像分割的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：

    ```shell
    $ hub serving start -m deeplabv3p_xception65_humanseg
    ```

    - 这样就完成了一个人像分割的服务化API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
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
  
    org_im = cv2.imread("/PATH/TO/IMAGE")
    # 发送HTTP请求
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/deeplabv3p_xception65_humanseg"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))# 保存图片
    mask =cv2.cvtColor(base64_to_cv2(r.json()["results"][0]['data']), cv2.COLOR_BGR2GRAY)
    rgba = np.concatenate((org_im, np.expand_dims(mask, axis=2)), axis=2)
    cv2.imwrite("segment_human_server.png", rgba)
    ```

## 五、更新历史

* 1.0.0

   初始发布

* 1.1.0

   提升预测性能

* 1.1.1

   修复预测后处理图像数据超过[0,255]范围

* 1.1.2

   移除 fluid api

  - ```shell
    $ hub install deeplabv3p_xception65_humanseg==1.1.2
    ```
