# dim_vgg16_matting

|Module Name|dim_vgg16_matting|
| :--- | :---: | 
|Category|Matting|
|Network|dim_vgg16|
|Dataset|Baidu self-built dataset|
|Support Fine-tuning|No|
|Module Size|164MB|
|Data Indicators|-|
|Latest update date|2021-12-03|


## I. Basic Information

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/144574288-28671577-8d5d-4b20-adb9-fe737015c841.jpg" width = "337" height = "505" hspace='10'/> 
    <img src="https://user-images.githubusercontent.com/35907364/144779164-47146d3a-58c9-4a38-b968-3530aa9a0137.png" width = "337" height = "505" hspace='10'/> 
    </p>

- ### Module Introduction

  - Mating is the technique of extracting foreground from an image by calculating its color and transparency. It is widely used in the film industry to replace background, image composition, and visual effects. Each pixel in the image will have a value that represents its foreground transparency, called Alpha. The set of all Alpha values in an image is called Alpha Matte. The part of the image covered by the mask can be extracted to complete foreground separation.


  
  - For more information, please refer to: [dim_vgg16_matting](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3/contrib/Matting)
  

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.2.0

    - paddlehub >= 2.1.0

    - paddleseg >= 2.3.0


- ### 2、Installation

    - ```shell
      $ hub install dim_vgg16_matting
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

    
## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run dim_vgg16_matting --input_path "/PATH/TO/IMAGE" --trimap_path "/PATH/TO/TRIMAP"
    ```
    
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example

    - ```python
      import paddlehub as hub
      import cv2

      model = hub.Module(name="dim_vgg16_matting")

      result = model.predict(image_list=["/PATH/TO/IMAGE"], trimap_list=["PATH/TO/TRIMAP"])
      print(result)
      ```
- ### 3、API

    - ```python
        def predict(self, 
                    image_list, 
                    trimap_list, 
                    visualization, 
                    save_path):
      ```

        - Prediction API for matting.

        - **Parameter**

            - image_list (list(str | numpy.ndarray)): Image path or image data, ndarray.shape is in the format \[H, W, C\]，BGR.
            - trimap_list(list(str | numpy.ndarray)): Trimap path or trimap data, ndarray.shape is in the format \[H, W]，Gray style. 
            - visualization (bool): Whether to save the recognition results as picture files, default is False.
            - save_path (str): Save path of images, "dim_vgg16_matting_output" by default.

        - **Return**

            - result (list(numpy.ndarray))：The list of model results.

 
## IV. Server Deployment

- PaddleHub Serving can deploy an online service of matting.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

    - ```shell
      $ hub serving start -m dim_vgg16_matting
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

    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))], 'trimaps':[cv2_to_base64(cv2.imread("/PATH/TO/TRIMAP"))]}

    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/dim_vgg16_matting"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    for image in r.json()["results"]['data']:
        data = base64_to_cv2(image)
        image_path =str(time.time()) + ".png"
        cv2.imwrite(image_path, data)
      ```

## V. Release Note

- 1.0.0

  First release
