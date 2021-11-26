# ssd_mobilenet_v1_pascal

|Module Name|ssd_mobilenet_v1_pascal|
| :--- | :---: |
|Category|object detection|
|Network|SSD|
|Dataset|PASCAL VOC|
|Fine-tuning supported or not|No|
|Module Size|24MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
     <p align="center">
     <img src="https://user-images.githubusercontent.com/22424850/131504887-d024c7e5-fc09-4d6b-92b8-4d0c965949d0.jpg"   width='50%' hspace='10'/>
     <br />
     </p>

- ### Module Introduction

  - Single Shot MultiBox Detector (SSD) 是一种单阶段的目标检测器.与两阶段的检测方法不同，单阶段目标检测并不进行区域推荐，而是直接从特征图回归出目标的边界框和分类概率.SSD 运用了这种单阶段检测的思想，并且对其进行改进：在不同尺度的特征图上检测对应尺度的目标.该PaddleHub Module的基网络为MobileNet-v1模型，在Pascal数据集上预训练得到，目前仅支持预测.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()

- ### 2、Installation

  - ```shell
    $ hub install ssd_mobilenet_v1_pascal
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run ssd_mobilenet_v1_pascal --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    object_detector = hub.Module(name="ssd_mobilenet_v1_pascal")
    result = object_detector.object_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = object_detector.object_detection((paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         output_dir='detection_result',
                         score_thresh=0.5,
                         visualization=True,
                         )
    ```

    - 预测API，检测输入图片中的所有目标的位置.

    - **Parameters**

      - paths (list[str]): image path;
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - output_dir (str): save path of images;
      - score\_thresh (float): 识别置信度的阈值；<br/>
      - visualization (bool): Whether to save the results as picture files;

      **NOTE:** choose one parameter to provide data from paths and images


    - **Return**

      - res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability
        - data (list): 检测结果，list的每一个元素为 dict，各字段为:
          - confidence (float): 识别的置信度
          - label (str): 标签
          - left (int): 边界框的左上角x坐标
          - top (int): 边界框的左上角y坐标
          - right (int): 边界框的右下角x坐标
          - bottom (int): 边界框的右下角y坐标
        - save\_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)

  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - 将模型保存到指定路径.

    - **Parameters**

      - dirname: 存在模型的目录名称； <br/>
      - model\_filename: 模型文件名称，默认为\_\_model\_\_； <br/>
      - params\_filename: Parameters文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)；<br/>
      - combined: 是否将Parameters保存到统一的一个文件中.


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of object detection.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m ssd_mobilenet_v1_pascal
    ```

  - The servitization API is now deployed and the default port number is 8866.

  - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result

  - ```python
    import requests
    import json
    import cv2
    import base64


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    # Send an HTTP request
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/ssd_mobilenet_v1_pascal"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

* 1.1.2

  Fix the problem of reading numpy

  - ```shell
    $ hub install ssd_mobilenet_v1_pascal==1.1.2
    ```
