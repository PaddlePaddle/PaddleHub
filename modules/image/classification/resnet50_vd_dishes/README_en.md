# resnet50_vd_dishes

|Module Name|resnet50_vd_dishes|
| :--- | :---: |
|Category|image classification|
|Network|ResNet50_vd|
|Dataset|Baidu Food Dataset|
|Fine-tuning supported or not|No|
|Module Size|158MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - ResNet proposed a residual unit to solve the problem of training an extremely deep network, and improved the prediction accuracy of models. ResNet-vd is a variant of ResNet. This module is based on ResNet-vd and can classify 8416 kinds of food.

<p align="center">
<img src="http://bj.bcebos.com/ibox-thumbnail98/77fa9b7003e4665867855b2b65216519?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2020-04-08T11%3A05%3A10Z%2F1800%2F%2F1df0ecb4a52adefeae240c9e2189e8032560333e399b3187ef1a76e4ffa5f19f" width = "800"  hspace='10'/> <br />
</p>

  - For more information, please refer to：[Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)

## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install resnet50_vd_dishes
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run resnet50_vd_dishes --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnet50_vd_dishes")
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
    $ hub serving start -m resnet50_vd_dishes
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
    url = "http://127.0.0.1:8866/predict/resnet50_vd_dishes"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install resnet50_vd_dishes==1.0.0
    ```
