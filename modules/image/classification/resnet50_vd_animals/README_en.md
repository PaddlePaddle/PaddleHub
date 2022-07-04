# resnet50_vd_animals

|Module Name|resnet50_vd_animals|
| :--- | :---: |
|Category |Image classification|
|Network|ResNet50_vd|
|Dataset|Baidu self-built dataset|
|Fine-tuning supported or not|No|
|Module Size|154MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I. Basic Information

- ### Application Effect Display

  - ResNet-vd is a variant of ResNet, which can be used for image classification and feature extraction. This module is trained by Baidu self-built animal data set and supports the classification and recognition of 7,978 animal species.
  - For more information, please refer to [ResNet-vd](https://arxiv.org/pdf/1812.01187.pdf)


## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0


- ### 2、Installation

  - ```shell
    $ hub install resnet50_vd_animals
    ```
  - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Command line Prediction

  - ```shell
    $ hub run resnet50_vd_animals --input_path "/PATH/TO/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)

- ### 2、Prediction Code Example

    - ```python
      import paddlehub as hub
      import cv2

      classifier = hub.Module(name="resnet50_vd_animals")

      result = classifier.classification(images=[cv2.imread('/PATH/TO/IMAGE')])
      # or
      # result = classifier.classification(paths=['/PATH/TO/IMAGE'])
      ```

- ### 3、API

    - ```python
      def get_expected_image_width()
      ```

        - Returns the preprocessed image width, which is 224.

    - ```python
      def get_expected_image_height()
      ```

        - Returns the preprocessed image height, which is 224.

    - ```python
      def get_pretrained_images_mean()
      ```

        - Returns the mean value of the preprocessed image, which is \[0.485, 0.456, 0.406\].

    - ```python
      def get_pretrained_images_std()
      ```

        - Return the standard deviation of the preprocessed image, which is \[0.229, 0.224, 0.225\].


    - ```python
      def classification(images=None,
                         paths=None,
                         batch_size=1,
                         use_gpu=False,
                         top_k=1):
      ```

        - **Parameter**

            * images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
            * paths (list\[str\]): image path;
            * batch\_size (int): batch size;
            * use\_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
            * top\_k (int): return the top k prediction results.

        - **Return**

            -   res (list\[dict\]): the list of classification results，key is the prediction label, value is the corresponding confidence.

    - ```python
      def save_inference_model(dirname,
                               model_filename=None,
                               params_filename=None,
                               combined=True)
      ```

        - Save the model to the specified path.

        - **Parameters**
            * dirname: Save path.
            * model\_filename: model file name，defalt is \_\_model\_\_
            * params\_filename: parameter file name，defalt is \_\_params\_\_(Only takes effect when `combined` is True)
            * combined: Whether to save the parameters to a unified file.



## IV. Server Deployment

- PaddleHub Serving can deploy an online service of animal classification.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        - ```shell
          $ hub serving start -m resnet50_vd_animals
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
        url = "http://127.0.0.1:8866/predict/resnet50_vd_animals"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))

        # print prediction results
        print(r.json()["results"])
        ```


## V. Release Note

- 1.0.0

  First release

* 1.0.1

  Remove fluid api

  - ```shell
    $ hub install resnet50_vd_animals==1.0.1
    ```
