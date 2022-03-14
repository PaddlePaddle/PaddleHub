# yolov3_darknet53_venus

|Module Name|yolov3_darknet53_venus|
| :--- | :---: |
|Category|object detection|
|Network|YOLOv3|
|Dataset|Baidu Detection Dataset|
|Fine-tuning supported or not|Yes|
|Module Size|501MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Module Introduction

  - YOLOv3 is a one-stage detector proposed by Joseph Redmon and Ali Farhadi, which can reach comparable accuracy but twice as fast as traditional methods. This module is based on YOLOv3, trained on Baidu Vehicle Dataset which consists of 170w pictures and 1000w+ boxes, improve the accuracy on 8 test datasets for average 5.36%, and can be used for vehicle detection.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub](../../../../docs/docs_en/get_start/installation.rst)  

- ### 2、Installation

  - ```shell
    $ hub install yolov3_darknet53_venus
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md) | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)

## III.Module API Prediction

- ### 1、API

  - ```python
    def context(trainable=True,
                pretrained=True,
                get_prediction=False)
    ```

    - Extract features, and do transfer learning

    - **Parameters**

      - trainable(bool): whether parameters trainable or not
      - pretrained (bool): whether load pretrained model or not
      - get\_prediction (bool): whether perform prediction

    - **Return**
      - inputs (dict): inputs, a dict, include two keys: "image" and "im\_size"
        - image (Variable): image variable
        - im\_size (Variable): image size
      - outputs (dict): model output
      - program for transfer learning

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         score_thresh=0.5,
                         visualization=True,
                         output_dir='detection_result')
    ```

    - Detection API, detect positions of all objects in image

    - **Parameters**

      - paths (list[str]): image path;
      - images (list\[numpy.ndarray\]): image data, ndarray.shape is in the format [H, W, C], BGR;
      - batch_size (int): the size of batch;
      - use_gpu (bool): use GPU or not; **set the CUDA_VISIBLE_DEVICES environment variable first if you are using GPU**
      - score\_thresh (float): confidence threshold；<br/>
      - visualization (bool): Whether to save the results as picture files;
      - output_dir (str): save path of images;

    - **Return**

      - res (list\[dict\]): classication results, each element in the list is dict, key is the label name, and value is the corresponding probability
        - data (list): detection results, each element in the list is dict
          - confidence (float): the confidence of the result
          - label (str): label
          - left (int): the upper left corner x coordinate of the detection box
          - top (int): the upper left corner y coordinate of the detection box
          - right (int): the lower right corner x coordinate of the detection box
          - bottom (int): the lower right corner y coordinate of the detection box
        - save\_path (str, optional): output path for saving results

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
    $ hub install yolov3_darknet53_venus==1.0.0
    ```
