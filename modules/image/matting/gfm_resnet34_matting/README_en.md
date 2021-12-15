# gfm_resnet34_matting

|Module Name|gfm_resnet34_matting|
| :--- | :---: | 
|Category|Image Matting|
|Network|gfm_resnet34|
|Dataset|AM-2k|
|Support Fine-tuning|No|
|Module Size|562MB|
|Data Indicators|SAD10.89|
|Latest update date|2021-12-03|


## I. Basic Information

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/145993777-9b69a85d-d31c-4743-8620-82b2a56ca1e7.jpg" width = "480" height = "350" hspace='10'/> 
    <img src="https://user-images.githubusercontent.com/35907364/145993809-b0fb4bae-2c64-4868-99fc-500f19343442.png" width = "480" height = "350" hspace='10'/> 
    </p>

- ### Module Introduction

  - Mating is the technique of extracting foreground from an image by calculating its color and transparency. It is widely used in the film industry to replace background, image composition, and visual effects. Each pixel in the image will have a value that represents its foreground transparency, called Alpha. The set of all Alpha values in an image is called Alpha Matte. The part of the image covered by the mask can be extracted to complete foreground separation.


  
  - For more information, please refer to: [gfm_resnet34_matting](https://github.com/JizhiziLi/GFM)
  

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.2.0

    - paddlehub >= 2.1.0

    - paddleseg >= 2.3.0


- ### 2、Installation

    - ```shell
      $ hub install gfm_resnet34_matting
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

    
## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run gfm_resnet34_matting --input_path "/PATH/TO/IMAGE"
    ```
    
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example

    - ```python
      import paddlehub as hub
      import cv2

      model = hub.Module(name="gfm_resnet34_matting")
      result = model.predict(["/PATH/TO/IMAGE"])
      print(result)

      ```
- ### 3、API

    - ```python
        def predict(self, 
                    image_list, 
                    visualization, 
                    save_path):
      ```

        - Prediction API for matting.

        - **Parameter**

            - image_list (list(str | numpy.ndarray)): Image path or image data, ndarray.shape is in the format \[H, W, C\]，BGR.
            - visualization (bool): Whether to save the recognition results as picture files, default is False.
            - save_path (str): Save path of images, "modnet_mobilenetv2_matting_output" by default.

        - **Return**

            - result (list(numpy.ndarray))：The list of model results.

 
## IV. Server Deployment

- PaddleHub Serving can deploy an online service of matting.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

    - ```shell
      $ hub serving start -m gfm_resnet34_matting
      ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

  - With a configured server, use the following lines of code to send the prediction request and obtain the result


    ```python
    import requests
    import json
    import cv2
    import base64
    import time
    import numpy as np

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')


    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}

    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/gfm_resnet34_matting"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    for image in r.json()["results"]['data']:
        data = base64_to_cv2(image)
        image_path =str(time.time()) + ".png"
        cv2.imwrite(image_path, data)
      ```

## V. Release Note

- 1.0.0

  First release
