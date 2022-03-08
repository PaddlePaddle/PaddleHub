# resnet50_vd_10w

|Module Name|resnet50_vd_10w|
| :--- | :---: |
|Category|image classification|
|Network|ResNet_vd|
|Dataset|Baidu Dataset|
|Fine-tuning supported or not|No|
|Module Size|92MB|
|Latest update date|-|
|Data indicators|-|


## I.Basic Information



- ### Module Introduction

  - ResNet proposed a residual unit to solve the problem of training an extremely deep network, and improved the prediction accuracy of models. ResNet-vd is a variant of ResNet. This module is based on ResNet_vd, trained on Baidu dataset(consists of 100 thousand classes, 40 million pairs of data), and can predict an image of size 224*224*3.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)


- ### 2、Installation

  - ```shell
    $ hub install resnet50_vd_10w
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、Prediction Code Example

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnet50_vd_10w")
    input_dict, output_dict, program = classifier.context(trainable=True)
    ```

- ### 2、API

  - ```python
    def context(trainable=True, pretrained=True)
    ```
    - **Parameters**
      - trainable (bool): whether parameters are trainable；<br/>
      - pretrained (bool): whether load the pre-trained model.

    - **Return**
      - inputs (dict): model inputs，key is 'image', value is the image tensor；<br/>
      - outputs (dict): model outputs，key is 'classification' and 'feature_map'，values：
        - classification (paddle.fluid.framework.Variable): classification result；
        - feature\_map (paddle.fluid.framework.Variable): feature map extracted by model.
      - context\_prog(fluid.Program): computation graph, used for transfer learning.



  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - **Parameters**
      - dirname: output dir for saving model; <br/>
      - model_filename: filename of model, default is \_\_model\_\_; <br/>
      - params_filename: filename of parameters，default is \_\_params\_\_(only effective when `combined` is True); <br/>
      - combined: whether save parameters into one file






## V.Release Note

* 1.0.0

  First release

  - ```shell
    $ hub install resnet50_vd_10w==1.0.0
    ```
