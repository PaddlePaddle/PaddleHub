# U2Net_Portrait

|模型名称|U2Net_Portrait|
| :--- | :---: | 
|类别|图像 - 图像生成|
|网络|U^2Net|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|254MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/07f73466f3294373965e06c141c4781992f447104a94471dadfabc1c3d920861"  width = "450" height = "300" hspace='10'/> <br />
    </p> 
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/c6ab02cf27414a5ba5921d9e6b079b487f6cda6026dc4d6dbca8f0167ad7cae3"  width = "450" height = "300" hspace='10'/> <br />
    </p> 
    

- ### 模型介绍

  - U2Net_Portrait 可以用于提取人脸的素描结果。


## 二、安装

- ### 1、环境依赖     

  - paddlepaddle >= 2.0.0    

  - paddlehub >= 2.0.0                            

- ### 2、安装

  - ```shell
    $ hub install U2Net_Portrait
    ```
  
## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run U2Net_Portrait --input_path "/PATH/TO/IMAGE"
    ```

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="U2Net_Portrait")
    result = model.Cartoon_GEN(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.Cartoon_GEN(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def Portrait_GEN(images=None,
                    paths=None,
                    scale=1,
                    batch_size=1,
                    output_dir='output',
                    face_detection=True,
                    visualization=False):
    ```

    - 人脸画像生成API。

    - **参数**

      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]；<br/>
      - paths (list\[str\]): 输入图像路径；<br/>
      - scale (float) : 缩放因子（与face_detection相关联)；<br/>
      - batch_size (int) : batch大小；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 output；<br/>
      - visualization (bool) : 是否将结果保存为图片文件；；<br/>

      **NOTE:** paths和images两个参数选择其一进行提供数据
    
    - **返回**
      - res (list\[numpy.ndarray\]): 输出图像数据，ndarray.shape 为 \[H, W, C\]


## 四、服务部署

- PaddleHub Serving可以部署一个在线人脸画像生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m U2Net_Portrait
    ```

  - 这样就完成了一个人脸画像生成的在线服务API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/U2Net_Portrait"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```


## 五、更新历史

* 1.0.0

  初始发布
   
  - ```shell
    $ hub install U2Net_Portrait==1.0.0
    ```