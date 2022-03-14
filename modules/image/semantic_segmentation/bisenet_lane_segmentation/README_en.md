# bisenet_lane_segmentation

|Module Name|bisenet_lane_segmentation|
| :--- | :---: | 
|Category|Image Segmentation|
|Network|bisenet|
|Dataset|TuSimple|
|Support Fine-tuning|No|
|Module Size|9.7MB|
|Data Indicators|ACC96.09%|
|Latest update date|2021-12-03|


## I. Basic Information

- ### Application Effect Display

  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/146115316-e9ed4220-8470-432f-b3f1-549d2bcdc845.jpg" /> 
    <img src="https://user-images.githubusercontent.com/35907364/146115396-a7d19290-6117-4831-bc35-4b14ae8f90bc.png" /> 
    </p>

- ### Module Introduction

  - Lane segmentation is a category of automatic driving algorithms, which can be used to assist vehicle positioning and decision-making. In the early days, there were lane detection methods based on traditional image processing, but with the evolution of technology, the scenes that lane detection tasks deal with More and more diversified, and more methods are currently seeking to detect the location of lane semantically. bisenet_lane_segmentation is a lightweight model for lane segmentation.


  
  - For more information, please refer to: [bisenet_lane_segmentation](https://github.com/PaddlePaddle/PaddleSeg)
  

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.2.0

    - paddlehub >= 2.1.0

    - paddleseg >= 2.3.0
    
    - Python >= 3.7+


- ### 2、Installation

    - ```shell
      $ hub install bisenet_lane_segmentation
      ```
      
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

    
## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run bisenet_lane_segmentation --input_path "/PATH/TO/IMAGE"
    ```
    
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example

    - ```python
      import paddlehub as hub
      import cv2

      model = hub.Module(name="bisenet_lane_segmentation")
      result = model.predict(image_list=["/PATH/TO/IMAGE"])
      print(result)

      ```
- ### 3、API

    - ```python
        def predict(self, 
                    image_list, 
                    visualization, 
                    save_path):
      ```

        - Prediction API for lane segmentation.

        - **Parameter**

            - image_list (list(str | numpy.ndarray)): Image path or image data, ndarray.shape is in the format \[H, W, C\]，BGR.
            - visualization (bool): Whether to save the recognition results as picture files, default is False.
            - save_path (str): Save path of images, "bisenet_lane_segmentation_output" by default.

        - **Return**

            - result (list(numpy.ndarray))：The list of model results.

 
## IV. Server Deployment

- PaddleHub Serving can deploy an online service of lane segmentation.

- ### Step 1: Start PaddleHub Serving

  - Run the startup command:

    - ```shell
      $ hub serving start -m bisenet_lane_segmentation
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

    import numpy as np


    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    def base64_to_cv2(b64str):
        data = base64.b64decode(b64str.encode('utf8'))
        data = np.fromstring(data, np.uint8)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return data

    org_im = cv2.imread('/PATH/TO/IMAGE')
    data = {'images':[cv2_to_base64(org_im)]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/bisenet_lane_segmentation"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))
    #print(r.json())
    mask = base64_to_cv2(r.json()["results"]['data'][0])
    print(mask)
    ```

## V. Release Note

- 1.0.0

  First release