# Photo2Cartoon

|模型名称|Photo2Cartoon|
| :--- | :---: | 
|类别|图像 - 图像生成|
|网络|U-GAT-IT|
|数据集|cartoon_data|
|是否支持Fine-tuning|否|
|模型大小|205MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201224164040624.jpg"  width = "450" height = "300" hspace='10'/> <br />
    </p> 
    
    

- ### 模型介绍

  - 本模型封装自[小视科技photo2cartoon项目的paddlepaddle版本](https://github.com/minivision-ai/photo2cartoon-paddle)。


## 二、安装

- ### 1、环境依赖     

  - paddlepaddle >= 2.0.0    

  - paddlehub >= 2.0.0                            

- ### 2、安装

  - ```shell
    $ hub install Photo2Cartoon
    ```
  
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run Photo2Cartoon --input_path "/PATH/TO/IMAGE"
    ```

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="Photo2Cartoon")
    result = model.Cartoon_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Cartoon_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def Cartoon_GEN(images=None,
                    paths=None,
                    batch_size=1,
                    output_dir='output',
                    visualization=False,
                    use_gpu=False):
    ```

    - 人像卡通化图像生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 输入图像路径；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - batch_size (int) : batch大小；<br/>     
      - visualization (bool) : 是否将结果保存为图片文件；；<br/>
      - use_gpu (bool) : 是否使用 GPU 进行推理。

      **NOTE:** paths和images两个参数选择其一进行提供数据
    
    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]


## 四、服务部署

- PaddleHub Serving可以部署一个在线人像卡通化图像生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m Photo2Cartoon
    ```

  - 这样就完成了一个人像卡通化图像生成的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/Photo2Cartoon"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布
   
  - ```shell
    $ hub install Photo2Cartoon==1.0.0
    ```