# prnet

|模型名称|prnet|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|PRN|
|数据集|300W-LP|
|是否支持Fine-tuning|否|
|模型大小|154MB|
|最新更新日期|2021-11-20|
|数据指标|-|


## 一、模型基本信息  

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/157190651-595b6964-97c5-4b0b-ac0a-c30c8520a972.png"  width = "450"  hspace='10'/>
    <br />
    输入原图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/142995636-dd5e1f0a-3810-4ae9-b680-4b2482858001.jpg"  width = "450" height = "300" hspace='10'/>
    <br />
    输入参考图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/157205282-89c9ace9-5fec-4112-ace1-edaebbfc30f8.png"  width = "450"  hspace='10'/>
    <br />
    输出图像
     <br />
    </p>

- ### 模型介绍

  - PRNet提出一种方法同时重建3D的脸部结构和脸部对齐，可应用于脸部对齐、3D脸重建、脸部纹理编辑等任务。该模块引入了脸部纹理编辑的功能，可以将参考图像的脸部纹理转移到原图像上。

  - 更多详情参考：[Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network](https://arxiv.org/pdf/1803.07835.pdf)



## 二、安装

- ### 1、环境依赖  
  - dlib
  - scikit-image

- ### 2、安装

  - ```shell
    $ hub install prnet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run prnet --source  "/PATH/TO/IMAGE1"  --ref "/PATH/TO/IMAGE2"
    ```
  - 通过命令行方式实现脸部纹理编辑的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    module = hub.Module(name="prnet")
    source_path = "/PATH/TO/IMAGE1"
    ref_path = "/PATH/TO/IMAGE2"
    module.face_swap(paths=[{'source':input_path, 'ref':ref_path}],
                    mode = 0,
                    output_dir='./swapping_result/',
                    use_gpu=True,
                    visualization=True)  
    ```

- ### 3、API

  - ```python
    def face_swap(self,
                images=None,
                paths=None,
                mode = 0,
                output_dir='./swapping_result/',
                use_gpu=False,
                visualization=True):
    ```
    - 脸部纹理编辑API，将参考图像的脸部纹理转移到原图像上。

    - **参数**
      - images (list[dict]): data of images, 每一个元素都为一个 dict，有关键字 source, ref, 相应取值为：
          - source (numpy.ndarray): 待转换的图片，shape 为 \[H, W, C\]，BGR格式；<br/>
          - ref (numpy.ndarray) : 参考图像，shape为 \[H, W, C\]，BGR格式；<br/>
      - paths (list[str]): paths to images, 每一个元素都为一个dict, 有关键字 source, ref, 相应取值为：
          - source (str): 待转换的图片的路径；<br/>
          - ref (str) : 参考图像的路径；<br/>
      - mode(int): option, 0表示改变局部纹理, 1表示改变整个脸；<br/>
      - output\_dir (str): 结果保存的路径； <br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - visualization(bool): 是否保存结果到本地文件夹

## 四、服务部署

- PaddleHub Serving可以部署一个在线图像风格转换服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m prnet
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
    data = {'images':[{'source': cv2_to_base64(cv2.imread("/PATH/TO/IMAGE1")), 'ref':cv2_to_base64(cv2.imread("/PATH/TO/IMAGE2"))}]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/prnet/"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install prnet==1.0.0
    ```
