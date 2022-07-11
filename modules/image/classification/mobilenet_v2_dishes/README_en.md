# mobilenet_v2_dishes

|Module Name|mobilenet_v2_dishes|
| :--- | :---: |
|Category|image classification|
|Network|MobileNet_v2|
|Dataset|Baidu food Dataset|
|Fine-tuning supported or not|No|
|Module Size|52MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - MobileNet is a light-weight convolution network. This module is trained on Baidu food dataset, and can classify 8416 kinds of food.

<p align="center">
<img src="http://bj.bcebos.com/ibox-thumbnail98/e7b22762cf42ab0e1e1fab6b8720938b?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2020-04-08T11%3A49%3A16Z%2F1800%2F%2Faf385f56da3c8ee1298588939d93533a72203c079ae1187affa2da555b9898ea" width = "800"  hspace='10'/> <br />
</p>

  - For more information, please refer to：[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install mobilenet_v2_dishes
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run mobilenet_v2_dishes --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="mobilenet_v2_dishes")
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
    $ hub serving start -m mobilenet_v2_dishes
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
    url = "http://127.0.0.1:8866/predict/mobilenet_v2_dishes"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install mobilenet_v2_dishes==1.0.0
    ```
