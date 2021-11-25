# resnet50_vd_imagenet_ssld

|Module Name|resnet50_vd_imagenet_ssld|
| :--- | :---: | 
|Category |Image classification|
|Network|ResNet_vd|
|Dataset|ImageNet-2012|
|Fine-tuning supported or notFine-tuning|Yes|
|Module Size|148MB|
|Data indicators|-|
|Latest update date|2021-02-26|


## I. Basic Information 

- ### Module Introduction

  - ResNet-vd is a variant of ResNet, which can be used for image classification and feature extraction.


## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $ hub install resnet50_vd_imagenet_ssld
      ```

    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)   

## III. Module API Prediction

- ### 1、Command line Prediction

    ```shell
    $ hub run resnet50_vd_imagenet_ssld --input_path "/PATH/TO/IMAGE" --top_k 5
    ```
- ### 2、Prediction Code Example

    ```python
    import paddle
    import paddlehub as hub

    if __name__ == '__main__':

        model = hub.Module(name='resnet50_vd_imagenet_ssld')
        result = model.predict(['/PATH/TO/IMAGE'])
    ```
- ### 3.Fine-tune and Encapsulation

    - After completing the installation of PaddlePaddle and PaddleHub, you can start using the user_guided_colorization model to fine-tune datasets such as [Flowers](../../docs/reference/datasets.md#class-hubdatasetsflowers) by excuting `python train.py`.

    - Steps:

        - Step1: Define the data preprocessing method
            - ```python
              import paddlehub.vision.transforms as T

              transforms = T.Compose([T.Resize((256, 256)),
                                    T.CenterCrop(224),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])],
                                    to_rgb=True)
              ```

             - `transforms`: The data enhancement module defines lots of data preprocessing methods. Users can replace the data preprocessing methods according to their needs.


        - Step2: Download the dataset

            - ```python
              from paddlehub.datasets import Flowers

              flowers = Flowers(transforms)

              flowers_validate = Flowers(transforms, mode='val')
              ```

                * `transforms`: data preprocessing methods.
                * `mode`: Select the data mode, the options are `train`, `test`, `val`. Default is `train`.
                * `hub.datasets.Flowers()` will be automatically downloaded from the network and decompressed to the `$HOME/.paddlehub/dataset` directory under the user directory.

        - Step3: Load the pre-trained model

            - ```python
              model = hub.Module(name="resnet50_vd_imagenet_ssld", label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"])
              ```
                * `name`: model name.
                * `label_list`: set the output classification category. Default is Imagenet2012 category.

        - Step4: Optimization strategy

            ```python
            optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
            trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt')

            trainer.train(flowers, epochs=100, batch_size=32, eval_dataset=flowers_validate, save_interval=1)
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

                  model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"], load_checkpoint='/PATH/TO/CHECKPOINT')
                  result = model.predict(['/PATH/TO/IMAGE'])
              ```

## IV. Server Deployment

- PaddleHub Serving can deploy an online service of classification.

- ### Step 1: Start PaddleHub Serving

    - Run the startup command:

        - ```shell
          $ hub serving start -m resnet50_vd_imagenet_ssld
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

        data = {'images':[cv2_to_base64(org_im)], 'top_k':2}
        headers = {"Content-type": "application/json"}
        url = "http://127.0.0.1:8866/predict/resnet50_vd_imagenet_ssld"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        data =r.json()["results"]['data']
        ```
## V. Release Note

* 1.0.0

  First release

* 1.1.0
    
  Upgrade to dynamic version
