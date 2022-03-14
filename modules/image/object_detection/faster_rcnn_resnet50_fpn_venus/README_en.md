# faster_rcnn_resnet50_fpn_venus

|Module Name|faster_rcnn_resnet50_fpn_venus|
| :--- | :---: |
|Category|object detection|
|Network|faster_rcnn|
|Dataset|Baidu Detection Dataset|
|Fine-tuning supported or not|Yes|
|Module Size|317MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Module Introduction

  - Faster_RCNN is a two-stage detector, it consists of feature extraction, proposal, classification and refinement processes. This module is trained on Baidu Detection Dataset, which contains 170w pictures and 1000w+ boxes, and improve the accuracy on 8 test datasets with average 2.06%. Besides, this module supports to fine-tune model, and can achieve faster convergence and better performance.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install faster_rcnn_resnet50_fpn_venus
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、API

  - ```python
    def context(num_classes=81,
                trainable=True,
                pretrained=True,
                phase='train')
    ```

    - Extract features, and do transfer learning

    - **Parameters**
      - num\_classes (int): number of classes；<br/>
      - trainable (bool): whether parameters trainable or not；<br/>
      - pretrained (bool): whether load pretrained model or not
      - get\_prediction (bool): optional, 'train' or 'predict'，'train' is used for training，'predict' used for prediction.

    - **Return**
      - inputs (dict): inputs, a dict：
        if phase is 'train', keys are：
          - image (Variable): image variable
          - im\_size (Variable): image size
          - im\_info (Variable): image information
          - gt\_class (Variable): box class
          - gt\_box (Variable): box coordination
          - is\_crowd (Variable): if multiple objects in box
        if phase 为 'predict'，keys are：
          - image (Variable): image variable
          - im\_size (Variable): image size
          - im\_info (Variable): image information
      - outputs (dict): model output
         if phase is 'train', keys are：
          - head_features (Variable): features extracted
          - rpn\_cls\_loss (Variable): classfication loss in box
          - rpn\_reg\_loss (Variable): regression loss in box
          - generate\_proposal\_labels (Variable): proposal labels
        if phase 为 'predict'，keys are：
          - head_features (Variable): features extracted
          - rois (Variable): roi
          - bbox\_out (Variable): prediction results
      - program for transfer learning

  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - Save model to specific path

    - **Parameters**

      - dirname: output dir for saving model
      - model\_filename: filename for saving model
      - params\_filename: filename for saving parameters
      - combined: whether save parameters into one file




## IV.Release Note

* 1.0.0

  First release
  - ```shell
    $ hub install faster_rcnn_resnet50_fpn_venus==1.0.0
    ```
