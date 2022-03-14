# falsr_a

|Module Name|falsr_a|
| :--- | :---: | 
|Category |Image editing|
|Network |falsr_a|
|Dataset|DIV2k|
|Fine-tuning supported or not|No|
|Module Size |8.9MB|
|Data indicators|PSNR37.82|
|Latest update date|2021-02-26|


## I. Basic Information 

- ### Application Effect Display
  
  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/133558583-0b7049db-ed1f-4a16-8676-f2141fcb3dee.png" width = "450" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130899031-a6f8c58a-5cb7-4105-b990-8cca5ae15368.png" width = "450" height = "300" hspace='10'/>
    </p>


- ### Module Introduction

  - Falsr_a is a lightweight super-resolution model based on "Accurate and Lightweight Super-Resolution with Neural Architecture Search". The model uses a multi-objective approach to deal with the over-segmentation problem, and uses an elastic search strategy based on a hybrid controller to improve the performance of the model. This model provides super resolution result with scale factor x2.

  - For more information, please refer to: [falsr_a](https://github.com/xiaomi-automl/FALSR)

## II. Installation

- ### 1、Environmental Dependence

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0


- ### 2、Installation

    - ```shell
      $ hub install falsr_a
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Command line Prediction

  - ```
    $ hub run falsr_a --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

  - ```python
    import cv2
    import paddlehub as hub

    sr_model = hub.Module(name='falsr_a')
    im = cv2.imread('/PATH/TO/IMAGE').astype('float32')
    res = sr_model.reconstruct(images=[im], visualization=True)
    print(res[0]['data'])
    sr_model.save_inference_model()
    ```

- ### 3、API

  - ```python
    def reconstruct(self,
                   images=None,
                   paths=None,
                   use_gpu=False,
                   visualization=False,
                   output_dir="falsr_a_output")
    ```

    - Prediction API.

    - **Parameter**

      * images (list\[numpy.ndarray\]): image data，ndarray.shape is in the format \[H, W, C\]，BGR.
      * paths (list\[str\]): image path.
      * use\_gpu (bool): use GPU or not. **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**.
      * visualization (bool): Whether to save the recognition results as picture files.
      * output\_dir (str): save path of images, "dcscn_output" by default.

    - **Return**
      * res (list\[dict\]): The list of model results, where each element is dict and each field is: 
        * save\_path (str, optional): Save path of the result, save_path is '' if no image is saved.
        * data (numpy.ndarray): result of super resolution.

  - ```python
    def save_inference_model(self,
                            dirname='falsr_a_save_model',
                            model_filename=None,
                            params_filename=None,
                            combined=False)
    ```

    - Save the model to the specified path.

    - **Parameters**

      * dirname: Save path.
      * model\_filename: model file name，defalt is \_\_model\_\_
      * params\_filename: parameter file name，defalt is \_\_params\_\_(Only takes effect when `combined` is True)
      * combined: Whether to save the parameters to a unified file.




## IV. Server Deployment

- PaddleHub Serving can deploy an online service of super resolution.


- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

      - ```shell
        $ hub serving start -m falsr_a
        ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result

      - ```python
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

        org_im = cv2.imread('/PATH/TO/IMAGE')
        data = {'images':[cv2_to_base64(org_im)]}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/falsr_a"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        sr = base64_to_cv2(r.json()["results"][0]['data'])
        cv2.imwrite('falsr_a_X2.png', sr)
        print("save image as falsr_a_X2.png")
        ```


## V. Release Note

- 1.0.0

  First release



