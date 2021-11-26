# mobilenet_v3_large_imagenet_ssld

|Module Name|mobilenet_v3_large_imagenet_ssld|
| :--- | :---: |
|Category|image classification|
|Network|Mobilenet_v3_large|
|Dataset|ImageNet-2012|
|Fine-tuning supported or not|No|
|Module Size|23MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - MobileNetV3是Google在2019年发布的新模型，作者通过结合NAS与NetAdapt进行搜索得到该网络结构，提供了Large和Small两个版本，分别适用于对资源不同要求的情况.对比于MobileNetV2，新的模型在速度和精度方面均有提升.该PaddleHubModule的模型结构为MobileNetV3 Large，基于ImageNet-2012数据集并采用PaddleClas提供的SSLD蒸馏方法训练得到，接受输入图片大小为224 x 224 x 3，支持finetune，也可以直接通过命令行或者Python接口进行预测.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()


- ### 2、Installation

  - ```shell
    $ hub install mobilenet_v3_large_imagenet_ssld
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run mobilenet_v3_large_imagenet_ssld --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="mobilenet_v3_large_imagenet_ssld")
    result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = classifier.classification(paths=['/PATH/TO/IMAGE'])
    ```

- ### 3、API


  - ```python
    def classification(images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       top_k=1):
    ```
    - classification API.
    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - top\_k (int): return the first k results

    - **Return**

      - res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability




## IV.Server Deployment

- PaddleHub Serving can deploy an online service of image classification.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m mobilenet_v3_large_imagenet_ssld
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
    url = "http://127.0.0.1:8866/predict/mobilenet_v3_large_imagenet_ssld"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install mobilenet_v3_large_imagenet_ssld==1.0.0
    ```
