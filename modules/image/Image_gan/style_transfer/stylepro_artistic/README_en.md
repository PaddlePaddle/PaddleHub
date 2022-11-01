# stylepro_artistic

|Module Name|stylepro_artistic|
| :--- | :---: |
|Category|image generation|
|Network|StyleProNet|
|Dataset|MS-COCO + WikiArt|
|Fine-tuning supported or not|No|
|Module Size|28MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Application Effect Display
  - Sample results：
    <p align="center">
    <img src="https://paddlehub.bj.bcebos.com/resources/style.png"  width='80%' hspace='10'/> <br />
    </p>

- ### Module Introduction

  - StyleProNet is a model for style transfer, which is light-weight and responds quickly. This module is based on StyleProNet, trained on WikiArt(MS-COCO) and WikiArt(style) datasets, and can be used for style transfer. For more information, please refer to [StyleProNet](https://arxiv.org/abs/2003.07694).


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0     | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install stylepro_artistic
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run stylepro_artistic --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_ch/tutorial/cmd_usage.rst)
- ### 2、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    stylepro_artistic = hub.Module(name="stylepro_artistic")
    result = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread('/PATH/TO/CONTENT_IMAGE'),
        'styles': [cv2.imread('/PATH/TO/STYLE_IMAGE')]
    }])

    # or
    # result = stylepro_artistic.style_transfer(
    #     paths=[{
    #         'content': '/PATH/TO/CONTENT_IMAGE',
    #         'styles': ['/PATH/TO/STYLE_IMAGE']
    #     }])
    ```

- ### 3、API

  - ```python
    def style_transfer(images=None,
                       paths=None,
                       alpha=1,
                       use_gpu=False,
                       visualization=False,
                       output_dir='transfer_result')
    ```

    - Style transfer API.

    - **Parameters**
      - images (list\[dict\]): each element is a dict，includes:
        - content (numpy.ndarray): input image array，shape is \[H, W, C\]，BGR format；<br/>
        - styles (list\[numpy.ndarray\]) : list of style image arrays，shape is \[H, W, C\]，BGR format；<br/>
        - weights (list\[float\], optioal) : weight for each style, if not set, each style has the same weight;<br/>
      - paths (list\[dict\]): each element is a dict，includes:
        - content (str): path for input image；<br/>
        - styles (list\[str\]) : paths for style images；<br/>
        - weights (list\[float\], optioal) :  weight for each style, if not set, each style has the same weight;<br/>
      - alpha (float) : alpha value，\[0, 1\] ，default is 1<br/>
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - visualization (bool): Whether to save the results as picture files;
      - output_dir (str): save path of images;

      **NOTE:** choose one parameter to provide data from paths and images

    - **Return**

      - res (list\[dict\]): results
        - path (str): path for input image
        - data (numpy.ndarray): output image


  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - Save model to specific path

    - **Parameters**

      - dirname: output dir for saving model
      - model\_filename: filename for saving model
      - params\_filename: filename for saving parameters
      - combined: whether save parameters into one file


## IV.Server Deployment

- PaddleHub Serving can deploy an online service of style transfer.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command：
  - ```shell
    $ hub serving start -m stylepro_artistic
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
    import numpy as np


    def cv2_to_base64(image):
      data = cv2.imencode('.jpg', image)[1]
      return base64.b64encode(data.tostring()).decode('utf8')

    def base64_to_cv2(b64str):
      data = base64.b64decode(b64str.encode('utf8'))
      data = np.fromstring(data, np.uint8)
      data = cv2.imdecode(data, cv2.IMREAD_COLOR)
      return data

    # Send an HTTP request
    data = {'images':[
    {
        'content':cv2_to_base64(cv2.imread('/PATH/TO/CONTENT_IMAGE')),
        'styles':[cv2_to_base64(cv2.imread('/PATH/TO/STYLE_IMAGE'))]
    }
    ]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/stylepro_artistic"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # print prediction results
    print(base64_to_cv2(r.json()["results"][0]['data']))
    ```


## V.Release Note

* 1.0.0

  First release

* 1.0.3

  Remove fluid api

  - ```shell
    $ hub install stylepro_artistic==1.0.3
    ```
