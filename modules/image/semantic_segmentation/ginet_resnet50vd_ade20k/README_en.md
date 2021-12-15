# ginet_resnet50vd_ade20k

|Module Name|ginet_resnet50vd_ade20k|
| :--- | :---: | 
|Category|Image Segmentation|
|Network|ginet_resnet50vd|
|Dataset|ADE20K|
|Fine-tuning supported or not|Yes|
|Module Size|214MB|
|Data indicators|-|
|Latest update date|2021-12-14|

## I. Basic Information 
  
- ### Application Effect Display
    - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/145947107-b6f87161-d824-4c21-b01d-594ad03e56de.jpg"  hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/145947270-1b8e0671-c5d3-4b61-b99e-0af27ccd9096.png" hspace='10'/>
    </p>

- ### Module Introduction

    - We will show how to use PaddleHub to finetune the pre-trained model and complete the prediction.
    - For more information, please refer to: [ginet](https://arxiv.org/pdf/2009.06160)

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install ginet_resnet50vd_ade20k
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  


## III. Module API Prediction

- ### 1、Prediction Code Example


    - ```python
      import cv2
      import paddle
      import paddlehub as hub

      if __name__ == '__main__':
          model = hub.Module(name='ginet_resnet50vd_ade20k')
          img = cv2.imread("/PATH/TO/IMAGE")
          result = model.predict(images=[img], visualization=True)
      ```

- ### 2.Fine-tune and Encapsulation

    - After completing the installation of PaddlePaddle and PaddleHub, you can start using the ginet_resnet50vd_ade20k model to fine-tune datasets such as OpticDiscSeg.

    - Steps:

         - Step1: Define the data preprocessing method

            - ```python
              from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize

              transform = Compose([Resize(target_size=(512, 512)), Normalize()])
              ```

            - `segmentation_transforms`: The data enhancement module defines lots of data preprocessing methods. Users can replace the data preprocessing methods according to their needs.

         - Step2: Download the dataset

            - ```python
              from paddlehub.datasets import OpticDiscSeg

              train_reader = OpticDiscSeg(transform, mode='train')

              ```
                * `transforms`: data preprocessing methods.

                * `mode`: Select the data mode, the options are `train`, `test`, `val`. Default is `train`.

                * Dataset preparation can be referred to [opticdiscseg.py](../../paddlehub/datasets/opticdiscseg.py)。`hub.datasets.OpticDiscSeg()`will be automatically downloaded from the network and decompressed to the `$HOME/.paddlehub/dataset` directory under the user directory.

        - Step3: Load the pre-trained model

            - ```python
              import paddlehub as hub

              model = hub.Module(name='ginet_resnet50vd_ade20k', num_classes=2, pretrained=None)
              ```
                - `name`: model name.
                - `load_checkpoint`: Whether to load the self-trained model, if it is None, load the provided parameters.

        - Step4:  Optimization strategy

            - ```python
              import paddle
              from paddlehub.finetune.trainer import Trainer

              scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=1000, power=0.9,  end_lr=0.0001)
              optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
              trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_seg', use_gpu=True)
              trainer.train(train_reader, epochs=10, batch_size=4, log_interval=10, save_interval=4)
              ```
             

    -  Model prediction

        - When Fine-tune is completed, the model with the best performance on the verification set will be saved in the `${CHECKPOINT_DIR}/best_model` directory. We use this model to make predictions. The `predict.py` script is as follows:

            ```python
            import paddle
            import cv2
            import paddlehub as hub

            if __name__ == '__main__':
                model = hub.Module(name='ginet_resnet50vd_ade20k', pretrained='/PATH/TO/CHECKPOINT')
                img = cv2.imread("/PATH/TO/IMAGE")
                model.predict(images=[img], visualization=True)
            ```

            - **Args**
                * `images`: Image path or ndarray data with format [H, W, C], BGR.
                * `visualization`: Whether to save the recognition results as picture files.
                * `save_path`: Save path of the result, default is 'seg_result'.


## IV. Server Deployment

- PaddleHub Serving can deploy an online service of image segmentation.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        - ```shell
          $ hub serving start -m ginet_resnet50vd_ade20k
          ```

    - The servitization API is now deployed and the default port number is 8866.

    - **NOTE:**  If GPU is used for prediction, set CUDA_VISIBLE_DEVICES environment variable before the service, otherwise it need not be set.

- ### Step 2: Send a predictive request

    - With a configured server, use the following lines of code to send the prediction request and obtain the result:

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
        url = "http://127.0.0.1:8866/predict/ginet_resnet50vd_ade20k"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        mask = base64_to_cv2(r.json()["results"][0])
        ```

## V. Release Note

- 1.0.0

  First release