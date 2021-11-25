# user_guided_colorization

|Module Name|user_guided_colorization|
| :--- | :---: | 
|Category |Image editing|
|Network| Local and Global Hints Network |
|Dataset|ILSVRC 2012|
|Fine-tuning supported or notFine-tuning|Yes|
|Module Size|131MB|
|Data indicators|-|
|Latest update date |2021-02-26|



## I. Basic Information 


- ### Application Effect Display
  
  - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/136653401-6644bd46-d280-4c15-8d48-680b7eb152cb.png" width = "300" height = "450" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/136648959-40493c9c-08ec-46cd-a2a2-5e2038dcbfa7.png" width = "300" height = "450" hspace='10'/>
    </p>

- ### Module Introduction

  - User_guided_colorization is a colorization model based on "Real-Time User-Guided Image Colorization with Learned Deep Priors"，this model uses pre-supplied coloring blocks to color the gray image.

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install user_guided_colorization
      ```

    - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)   

## III. Module API Prediction

- ### 1、Command line Prediction

    ```shell
    $ hub run user_guided_colorization --input_path "/PATH/TO/IMAGE"
    ```

     - If you want to call the Hub module through the command line, please refer to: [PaddleHub Command Line Instruction](../../../../docs/docs_en/tutorial/cmd_usage.rst)
- ### 2、Prediction Code Example

    ```python
    import paddle
    import paddlehub as hub

    if __name__ == '__main__':

        model = hub.Module(name='user_guided_colorization')
        model.set_config(prob=0.1)
        result = model.predict(images=['/PATH/TO/IMAGE'])
    ```
- ### 3.Fine-tune and Encapsulation

    - After completing the installation of PaddlePaddle and PaddleHub, you can start using the user_guided_colorization model to fine-tune datasets such as [Canvas](../../docs/reference/datasets.md#class-hubdatasetsCanvas) by executing `python train.py`.

    - Steps:

        - Step1: Define the data preprocessing method

            - ```python
              import paddlehub.vision.transforms as T

              transform = T.Compose([T.Resize((256, 256), interpolation='NEAREST'),
                       T.RandomPaddingCrop(crop_size=176),
                       T.RGB2LAB()], to_rgb=True)
              ```

              - `transforms`: The data enhancement module defines lots of data preprocessing methods. Users can replace the data preprocessing methods according to their needs.

        - Step2: Download the dataset
            - ```python
              from paddlehub.datasets import Canvas

              color_set = Canvas(transform=transform, mode='train')
              ```

                * `transforms`: Data preprocessing methods.
                * `mode`: Select the data mode, the options are `train`, `test`, `val`. Default is `train`.
                * `hub.datasets.Canvas()`: The dataset will be automatically downloaded from the network and decompressed to the `$HOME/.paddlehub/dataset` directory under the user directory.


        - Step3: Load the pre-trained model

            - ```python
              model = hub.Module(name='user_guided_colorization', load_checkpoint=None)
              model.set_config(classification=True, prob=1)
              ```
                * `name`: Model name.
                * `load_checkpoint`: Whether to load the self-trained model, if it is None, load the provided parameters.
                * `classification`: The model is trained by two mode. At the beginning, `classification` is set to True, which is used for shallow network training. In the later stage of training, set `classification` to False, which is used to train the output layer of the network.
                * `prob`: The probability that a priori color block is not added to each input image, the default is 1, that is, no prior color block is added. For example, when `prob` is set to 0.9, the probability that there are two a priori color blocks on a picture is(1-0.9)*(1-0.9)*0.9=0.009.

        - Step4: Optimization strategy

            ```python
            optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
            trainer = Trainer(model, optimizer, checkpoint_dir='img_colorization_ckpt_cls_1')
            trainer.train(color_set, epochs=201, batch_size=25, eval_dataset=color_set, log_interval=10, save_interval=10)
            ```


            - Run configuration

            - `Trainer` mainly control the training of Fine-tune, including the following controllable parameters:

                * `model`: Optimized model.
                * `optimizer`: Optimizer selection.
                * `use_vdl`: Whether to use vdl to visualize the training process.
                * `checkpoint_dir`: The storage address of the model parameters.
                * `compare_metrics`: The measurement index of the optimal model.

            - `trainer.train` mainly control the specific training process, including the following controllable parameters:

                * `train_dataset`: Training dataset.
                * `epochs`: Epochs of training process.
                * `batch_size`: Batch size.
                * `num_workers`: Number of workers.
                * `eval_dataset`: Validation dataset.
                * `log_interval`:The interval for printing logs.
                * `save_interval`: The interval for saving model parameters.

    - Model prediction

        -   When Fine-tune is completed, the model with the best performance on the verification set will be saved in the `${CHECKPOINT_DIR}/best_model` directory. We use this model to make predictions. The `predict.py` script is as follows:

            - ```python
              import paddle
              import paddlehub as hub

              if __name__ == '__main__':
                  model = hub.Module(name='user_guided_colorization', load_checkpoint='/PATH/TO/CHECKPOINT')
                  model.set_config(prob=0.1)
                  result = model.predict(images=['/PATH/TO/IMAGE'])
              ```


            - **NOTE:** If you want to get the oil painting style, please download the parameter file [Canvas colorization](https://paddlehub.bj.bcebos.com/dygraph/models/canvas_rc.pdparams)

## IV. Server Deployment

- PaddleHub Serving can deploy an online service of colorization.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

      - ```shell
        $ hub serving start -m user_guided_colorization
        ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:** If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

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
        org_im = cv2.imread('/PATH/TO/IMAGE')
        data = {'images':[cv2_to_base64(org_im)]}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/user_guided_colorization"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        data = base64_to_cv2(r.json()["results"]['data'][0]['fake_reg'])
        cv2.imwrite('color.png', data)
        ```


## V. Release Note

* 1.0.0

  First release
