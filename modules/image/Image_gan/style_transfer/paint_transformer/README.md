# paint_transformer

|模型名称|paint_transformer|
| :--- | :---: |
|类别|图像 - 风格转换|
|网络|Paint Transformer|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|77MB|
|最新更新日期|2021-12-07|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/145002878-ffdeea71-8ff4-48cc-88d0-fba1aa1dce4b.jpg"  width = "40%"  hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/145002301-97c45887-cb2e-4a06-9d00-07b74080effa.png"  width = "40%"  hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - 该模型可以实现图像油画风格的转换。
  - 更多详情参考：[Paint Transformer: Feed Forward Neural Painting with Stroke Prediction](https://github.com/wzmsltw/PaintTransformer)



## 二、安装

- ### 1、环境依赖  
  - ppgan

- ### 2、安装

  - ```shell
    $ hub install paint_transformer
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    # Read from a file
    $ hub run paint_transformer --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现风格转换模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="paint_transformer")
    input_path = ["/PATH/TO/IMAGE"]
    # Read from a file
    module.style_transfer(paths=input_path, output_dir='./transfer_result/', use_gpu=True)  
    ```

- ### 3、API

  - ```python
    style_transfer(images=None, paths=None, output_dir='./transfer_result/', use_gpu=False, need_animation=False, visualization=True):
    ```
    - 油画风格转换API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 图片的路径；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - need_animation(bool): 是否保存中间结果形成动画
      - visualization(bool): 是否保存结果到本地文件夹


## 四、服务部署

- PaddleHub Serving可以部署一个在线油画风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m paint_transformer
    ```

  - 这样就完成了一个油画风格转换的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/paint_transformer"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install paint_transformer==1.0.0
    ```
