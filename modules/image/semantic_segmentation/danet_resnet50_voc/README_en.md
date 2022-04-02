# danet_resnet50_voc

|Module Name|danet_resnet50_voc|
| :--- | :---: | 
|Category|Image Segmentation|
|Network|danet_resnet50vd|
|Dataset|PascalVOC2012|
|Fine-tuning supported or not|Yes|
|Module Size|273MB|
|Data indicators|-|
|Latest update date|2022-03-22|

## I. Basic Information 
  
- ### Application Effect Display
    - Sample results:
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/159212097-443a5a65-2f2e-4126-9c07-d7c3c220e55f.jpg"  width = "420" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/159212375-52e123af-4699-4c25-8f50-4240bbb714b4.png" width = "420" height = "505" hspace='10'/>
    </p>

- ### Module Introduction

    - We will show how to use PaddleHub to finetune the pre-trained model and complete the prediction.
    - For more information, please refer to: [danet](https://arxiv.org/pdf/1809.02983.pdf)

## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install danet_resnet50_voc
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
          model = hub.Module(name='danet_resnet50_voc')
          img = cv2.imread("/PATH/TO/IMAGE")
          result = model.predict(images=[img], visualization=True)
      ```

- ### 2.Fine-tune and Encapsulation

    - After completing the installation of PaddlePaddle and PaddleHub, you can start using the danet_resnet50_voc model to fine-tune datasets such as OpticDiscSeg.

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

              model = hub.Module(name='danet_resnet50_voc', num_classes=2, pretrained=None)
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
                model = hub.Module(name='danet_resnet50_voc', pretrained='/PATH/TO/CHECKPOINT')
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
          $ hub serving start -m danet_resnet50_voc
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
        url = "http://127.0.0.1:8866/predict/danet_resnet50_voc"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        mask = base64_to_cv2(r.json()["results"][0])
        ```

## V. Release Note

- 1.0.0

  First release
