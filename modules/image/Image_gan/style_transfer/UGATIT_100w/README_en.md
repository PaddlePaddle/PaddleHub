# UGATIT_100w

|Module Name|UGATIT_100w|
| :--- | :---: |
|Category|image generation|
|Network|U-GAT-IT|
|Dataset|selfie2anime|
|Fine-tuning supported or not|No|
|Module Size|41MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://ai-studio-static-online.cdn.bcebos.com/d130fabd8bd34e53b2f942b3766eb6bbd3c19c0676d04abfbd5cc4b83b66f8b6"  height='80%' hspace='10'/>
    <br />
    Input image
    <br />
    <img src="https://ai-studio-static-online.cdn.bcebos.com/8538af03b3f14b1884fcf4eec48965baf939e35a783d40129085102057438c77"   height='80%' hspace='10'/>
    <br />
    Output image
    <br />
    </p>


- ### Module Introduction

  - UGATIT is a model for style transfer. This module can be used to transfer a face image to cartoon style. For more information, please refer to [UGATIT-Paddle Project](https://github.com/miraiwk/UGATIT-paddle).


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.8.0  

  - paddlehub >= 1.8.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)

- ### 2、Installation

  - ```shell
    $ hub install UGATIT_100w
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    model = hub.Module(name="UGATIT_100w")
    result = model.style_transfer(images=[cv2.imread('/PATH/TO/IMAGE')])
    # or
    # result = model.style_transfer(paths=['/PATH/TO/IMAGE'])
    ```

- ### 2、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       batch_size=1,
                       output_dir='output',
                       visualization=False)
    ```

    - Style transfer API.

    - **Parameters**

      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - paths (list[str]): image path;
      - batch_size (int): the size of batch;
      - visualization (bool): Whether to save the results as picture files;
      - output_dir (str): save path of images;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**
      - res (list\[numpy.ndarray\]): result list，ndarray.shape is  \[H, W, C\]


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of style transfer.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m UGATIT_100w
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
    url = "http://127.0.0.1:8866/predict/UGATIT_100w"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(r.json()["results"])
    ```


## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install UGATIT_100w==1.0.0
    ```
