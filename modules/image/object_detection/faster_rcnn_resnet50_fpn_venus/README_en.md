# faster_rcnn_resnet50_fpn_venus

|Module Name|faster_rcnn_resnet50_fpn_venus|
| :--- | :---: |
|Category|object detection|
|Network|faster_rcnn|
|Dataset|百度自建Dataset|
|Fine-tuning supported or not|Yes|
|Module Size|317MB|
|Latest update date|2021-02-26|
|Data indicators|-|


## I.Basic Information

- ### Module Introduction

  - Faster_RCNN是两阶段目标检测器，对图像生成候选区域、提取特征、判别特征类别并修正候选框位置.Faster_RCNN整体网络可以分为4个部分，一是ResNet-50作为基础卷积层，二是区域生成网络，三是Rol Align，四是检测层.该PaddleHub Module是由800+tag,170w图片，1000w+检测框训练的大规模通用检测模型，在8个数据集上MAP平均提升2.06%，iou=0.5的准确率平均提升1.78%.对比于其他通用检测模型，使用该Module进行finetune，可以更快收敛，达到较优效果.


## II.Installation

- ### 1、Environmental Dependence  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [How to install PaddleHub]()  

- ### 2、Installation

  - ```shell
    $ hub install faster_rcnn_resnet50_fpn_venus
    ```
  - In case of any problems during installation, please refer to: [Windows_Quickstart]() | [Linux_Quickstart]() | [Mac_Quickstart]()

## III.Module API Prediction

- ### 1、API

  - ```python
    def context(num_classes=81,
                trainable=True,
                pretrained=True,
                phase='train')
    ```

    - 提取特征，用于迁移学习.

    - **Parameters**
      - num\_classes (int): 类别数；<br/>
      - trainable (bool): Parameters是否可训练；<br/>
      - pretrained (bool): 是否加载预训练模型；<br/>
      - get\_prediction (bool): 可选值为 'train'/'predict'，'train' 用于训练，'predict' 用于预测.

    - **Return**
      - inputs (dict): 模型的输入，相应的取值为：
        当phase为'train'时，包含：
          - image (Variable): 图像变量
          - im\_size (Variable): 图像的尺寸
          - im\_info (Variable): 图像缩放信息
          - gt\_class (Variable): 检测框类别
          - gt\_box (Variable): 检测框坐标
          - is\_crowd (Variable): 单个框内是否包含多个物体
        当 phase 为 'predict'时，包含：
          - image (Variable): 图像变量
          - im\_size (Variable): 图像的尺寸
          - im\_info (Variable): 图像缩放信息
      - outputs (dict): 模型的输出，响应的取值为：
        当 phase 为 'train'时，包含：
          - head_features (Variable): 所提取的特征
          - rpn\_cls\_loss (Variable): 检测框分类损失
          - rpn\_reg\_loss (Variable): 检测框回归损失
          - generate\_proposal\_labels (Variable): 图像信息
        当 phase 为 'predict'时，包含：
          - head_features (Variable): 所提取的特征
          - rois (Variable): 提取的roi
          - bbox\_out (Variable): 预测结果
      - context\_prog (Program): 用于迁移学习的 Program

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
