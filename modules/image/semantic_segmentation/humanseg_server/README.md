# humanseg_server

|模型名称|humanseg_server|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|hrnet|
|数据集|百度自建数据集|
|是否支持Fine-tuning|否|
|模型大小|159MB|
|指标|-|
|最新更新日期|2021-02-26|



## 一、模型基本信息

- ### 应用效果展示
  
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130913092-312a5f37-842e-4fd0-8db4-5f853fd8419f.jpg" width = "337" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130915531-bd4b2294-47e4-47e1-b9d3-3c1fa8b90f8f.png" width = "337" height = "505" hspace='10'/>
    </p>

- ### 模型介绍

    - HumanSeg-server使用百度自建数据集进行训练，可用于人像分割，支持任意大小的图片输入。

    - 更多详情请参考：[humanseg_server](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/HumanSeg)

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >=2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install humanseg_server
      ```
      
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

    ```
    hub run humanseg_server --input_path "/PATH/TO/IMAGE"
    ```
- ### 2、预测代码示例

    - 图片分割及视频分割代码示例：

    ```python
    import cv2
    import paddlehub as hub

    human_seg = hub.Module(name='humanseg_server')
    im = cv2.imread('/PATH/TO/IMAGE')
    #visualization=True可以用于查看人像分割图片效果，可设置为False提升运行速度。
    res = human_seg.segment(images=[im],visualization=True)
    print(res[0]['data'])
    human_seg.video_segment('/PATH/TO/VIDEO')
    human_seg.save_inference_model('/PATH/TO/SAVE/MODEL')

    ```
    - 视频流预测代码示例：

    ```python
    import cv2
    import numpy as np
    import paddlehub as hub

    human_seg = hub.Module(name='humanseg_server')
    cap_video = cv2.VideoCapture('\PATH\TO\VIDEO')
    fps = cap_video.get(cv2.CAP_PROP_FPS)
    save_path = 'humanseg_server_video.avi'
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
    prev_gray = None
    prev_cfd = None
    while cap_video.isOpened():
        ret, frame_org = cap_video.read()
        if ret:
            [img_matting, prev_gray, prev_cfd] = human_seg.video_stream_segment(frame_org=frame_org, frame_id=cap_video.get(1), prev_gray=prev_gray, prev_cfd=prev_cfd)
            img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
            bg_im = np.ones_like(img_matting) * 255
            comb = (img_matting * frame_org + (1 - img_matting) * bg_im).astype(np.uint8)
            cap_out.write(comb)
        else:
            break

    cap_video.release()
    cap_out.release()

    ```

- ### 3、API

    ```python
    def segment(images=None,
                paths=None,
                batch_size=1,
                use_gpu=False,
                visualization=False,
                output_dir='humanseg_server_output')
    ```

    - 预测API，用于人像分割。

    - **参数**

        * images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
        * paths (list\[str\]): 图片的路径；
        * batch\_size (int): batch 的大小；
        * use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；
        * visualization (bool): 是否将识别结果保存为图片文件；
        * output\_dir (str): 图片的保存路径。

    - **返回**

        * res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 'save\_path', 'data'，对应的取值为：
            * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
            * data (numpy.ndarray): 人像分割结果，仅包含Alpha通道，取值为0-255 (0为全透明，255为不透明)，也即取值越大的像素点越可能为人体，取值越小的像素点越可能为背景。


    ```python
    def video_stream_segment(self,
                            frame_org,
                            frame_id,
                            prev_gray,
                            prev_cfd,
                            use_gpu=False):
    ```

    -  预测API，用于逐帧对视频人像分割。

    - **参数**

        * frame_org (numpy.ndarray): 单帧图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
        * frame_id (int): 当前帧的编号；
        * prev_gray (numpy.ndarray): 前一帧输入网络图像的灰度图；
        * prev_cfd (numpy.ndarray): 前一帧光流追踪图和预测结果融合图
        * use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；


    -  **返回**

        * img_matting (numpy.ndarray): 人像分割结果，仅包含Alpha通道，取值为0-1 (0为全透明，1为不透明)。
        * cur_gray (numpy.ndarray): 当前帧输入网络图像的灰度图；
        * optflow_map (numpy.ndarray): 当前帧光流追踪图和预测结果融合图


    ```python
    def video_segment(self,
                      video_path=None,
                      use_gpu=False,
                      save_dir='humanseg_server_video_result'):
    ```

    -  预测API，用于视频人像分割。

    - **参数**

        * video\_path (str): 待分割视频路径。若为None，则从本地摄像头获取视频，并弹出窗口显示在线分割结果。
        * use\_gpu (bool): 是否使用 GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置；
        * save\_dir (str): 视频保存路径，仅在video\_path不为None时启用，保存离线视频处理结果。


    ```python
    def save_inference_model(dirname='humanseg_server_model',
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```

    -  将模型保存到指定路径。

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
    $ hub serving start -m humanseg_server
    ```

    - 这样就完成了一个人像分割的服务化API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

    - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

        ```python
        import requests
        import json
        import base64

        import cv2
        import numpy as np

        def cv2_to_base64(image):
            data = cv2.imencode('.jpg', image)[1]
            return base64.b64encode(data.tostring()).decode('utf8')
        def base64_to_cv2(b64str):
            data = base64.b64decode(b64str.encode('utf8'))
            data = np.fromstring(data, np.uint8)
            data = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return data

        # 发送HTTP请求
        org_im = cv2.imread('/PATH/TO/IMAGE')
        data = {'images':[cv2_to_base64(org_im)]}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/humanseg_server"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # 保存图片
        mask =cv2.cvtColor(base64_to_cv2(r.json()["results"][0]['data']), cv2.COLOR_BGR2GRAY)
        rgba = np.concatenate((org_im, np.expand_dims(mask, axis=2)), axis=2)
        cv2.imwrite("segment_human_server.png", rgba)
        ```


## 五、更新历史

* 1.0.0

    初始发布
* 1.1.0
    
    新增视频人像分割接口

    新增视频流人像分割接口
* 1.1.1

   修复cudnn为8.0.4显存泄露问题
