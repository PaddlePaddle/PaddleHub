# pp-tinypose

|模型名称|pp-tinypose|
| :--- | :---: |
|类别|图像-关键点检测|
|网络|PicoDet + tinypose|
|数据集|COCO + AI Challenger|
|是否支持Fine-tuning|否|
|模型大小|125M|
|最新更新日期|2022-05-20|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
<p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/169768593-9fcf729a-458e-4bb1-bb3c-b005ff7bcec2.jpg"   hspace='10'/>
    <br />
    输入图像
    <br />
    <img src="https://user-images.githubusercontent.com/22424850/170029768-3c60def2-7c87-4e8a-98bc-1bbc912204e7.jpg"   hspace='10'/>
    <br />
    输出图像

- ### 模型介绍

  - PP-TinyPose是PaddleDetecion针对移动端设备优化的实时关键点检测模型，可流畅地在移动端设备上执行多人姿态估计任务。借助PaddleDetecion自研的优秀轻量级检测模型PicoDet以及轻量级姿态估计任务骨干网络Tinypose, 结合多种策略有效平衡了模型的速度和精度表现。

  - 更多详情参考：[PP-TinyPose](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/keypoint/tiny_pose)。



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.2

  - paddlehub >= 2.2   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install pp-tinypose
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run pp-tinypose --input_path "/PATH/TO/IMAGE" --visualization True --use_gpu
    ```
  - 通过命令行方式实现关键点检测模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="pp-tinypose")
    result = model.predict('/PATH/TO/IMAGE', save_path='pp_tinypose_output', visualization=True, use_gpu=True)
    ```

- ### 3、API


  - ```python
    def predict(self, img: Union[str, np.ndarray], save_path: str = "pp_tinypose_output", visualization: bool = True, use_gpu = False)
    ```

    - 预测API，识别输入图片中的所有人肢体关键点。

    - **参数**

      - img (numpy.ndarray|str): 图片数据，使用图片路径或者输入numpy.ndarray，BGR格式；
      - save_path (str): 图片保存路径， 默认为'pp_tinypose_output'；
      - visualization (bool): 是否将识别结果保存为图片文件；
      - use_gpu: 是否使用gpu；
    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表元素依然为列表，存的内容为[图像名称，检测框，关键点]。


## 四、服务部署

- PaddleHub Serving 可以部署一个关键点检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m pp-tinypose
    ```

  - 这样就完成了一个关键点检测的服务化API的部署，默认端口号为8866。

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
    url = "http://127.0.0.1:8866/predict/pp-tinypose"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    ```

## 五、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install pp-tinypose==1.0.0
    ```
