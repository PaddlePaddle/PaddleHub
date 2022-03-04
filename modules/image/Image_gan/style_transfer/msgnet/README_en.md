# msgnet

|Module Name|msgnet|
| :--- | :---: | 
|Category|Image editing|
|Network|msgnet|
|Dataset|COCO2014|
|Fine-tuning supported or not|Yes|
|Module Size|68MB|
|Data indicators|-|
|Latest update date|2021-07-29|


## I. Basic Information 
  
- ### Application Effect Display
    - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/130910325-d72f34b2-d567-4e77-bb60-35148864301e.jpg" width = "450" height = "300" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/130910195-9433e4a7-3596-4677-85d2-2ffc16939597.png" width = "450" height = "300" hspace='10'/>
    </p>

- ### Module Introduction

    - Msgnet is a style transfer model. We will show how to use PaddleHub to finetune the pre-trained model and complete the prediction.
    - For more information, please refer to [msgnet](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer)

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install msgnet
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Command line Prediction

  - ```
    $ hub run msgnet --input_path "/PATH/TO/ORIGIN/IMAGE" --style_path "/PATH/TO/STYLE/IMAGE"
    ```
  - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)


- ### 2、Prediction Code Example

    -  ```python
        import paddle
        import paddlehub as hub

        if __name__ == '__main__':
            model = hub.Module(name='msgnet')
            result = model.predict(origin=["/PATH/TO/ORIGIN/IMAGE"], style="/PATH/TO/STYLE/IMAGE", visualization=True, save_path ="/PATH/TO/SAVE/IMAGE")
        ```

- ### 3.Fine-tune and Encapsulation

    - After completing the installation of PaddlePaddle and PaddleHub, you can start using the msgnet model to fine-tune datasets such as [MiniCOCO](../../docs/reference/datasets.md#class-hubdatasetsMiniCOCO) by executing `python train.py`.

    - Steps:

        - Step1: Define the data preprocessing method

            - ```python
              import paddlehub.vision.transforms as T

              transform = T.Compose([T.Resize((256, 256), interpolation='LINEAR')])
              ```

            - `transforms` The data enhancement module defines lots of data preprocessing methods. Users can replace the data preprocessing methods according to their needs.

        - Step2: Download the dataset
            - ```python
              from paddlehub.datasets.minicoco import MiniCOCO

              styledata = MiniCOCO(transform=transform, mode='train')

              ```
                * `transforms`: data preprocessing methods.
                * `mode`: Select the data mode, the options are `train`, `test`, `val`. Default is `train`.

                - Dataset preparation can be referred to [minicoco.py](../../paddlehub/datasets/minicoco.py). `hub.datasets.MiniCOCO()` will be automatically downloaded from the network and decompressed to the `$HOME/.paddlehub/dataset` directory under the user directory.

        - Step3: Load the pre-trained model

            - ```python
              model = hub.Module(name='msgnet', load_checkpoint=None)
              ```
                * `name`: model name.
                * `load_checkpoint`: Whether to load the self-trained model, if it is None, load the provided parameters.

        - Step4: Optimization strategy

            - ```python
              optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
              trainer = Trainer(model, optimizer, checkpoint_dir='test_style_ckpt')
              trainer.train(styledata, epochs=101, batch_size=4, eval_dataset=styledata, log_interval=10, save_interval=10)
              ```


    - Model prediction

        -   When Fine-tune is completed, the model with the best performance on the verification set will be saved in the `${CHECKPOINT_DIR}/best_model` directory. We use this model to make predictions. The `predict.py` script is as follows:
            -   ```python
                import paddle
                import paddlehub as hub

                if __name__ == '__main__':
                    model = hub.Module(name='msgnet', load_checkpoint="/PATH/TO/CHECKPOINT")
                    result = model.predict(origin=["/PATH/TO/ORIGIN/IMAGE"], style="/PATH/TO/STYLE/IMAGE", visualization=True, save_path ="/PATH/TO/SAVE/IMAGE")
                ```

                - **Parameters**
                    * `origin`: Image path or ndarray data with format [H, W, C], BGR.
                    * `style`: Style image path.
                    * `visualization`: Whether to save the recognition results as picture files.
                    * `save_path`: Save path of the result, default is 'style_tranfer'.


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of style transfer.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        - ```shell
        $ hub serving start -m msgnet
        ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.


- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result:

        -   ```python
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
            org_im = cv2.imread('/PATH/TO/ORIGIN/IMAGE')
            style_im = cv2.imread('/PATH/TO/STYLE/IMAGE')
            data = {'images':[[cv2_to_base64(org_im)], cv2_to_base64(style_im)]}
            headers = {"Content-type": "application/json"}
            url = "http://127.0.0.1:8866/predict/msgnet"
            r = requests.post(url=url, headers=headers, data=json.dumps(data))
            data = base64_to_cv2(r.json()["results"]['data'][0])
            cv2.imwrite('style.png', data)
            ```

## V. Release Note

- 1.0.0

  First release
