# yolov3_darknet53_venus

|模型名称|yolov3_darknet53_venus|
| :--- | :---: |
|类别|图像 - 目标检测|
|网络|YOLOv3|
|数据集|百度自建数据集|
|是否支持Fine-tuning|是|
|模型大小|501MB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 模型介绍

  - YOLOv3是由Joseph Redmon和Ali Farhadi提出的单阶段检测器, 该检测器与达到同样精度的传统目标检测方法相比，推断速度能达到接近两倍。 YOLOv3将输入图像划分格子，并对每个格子预测bounding box。YOLOv3的loss函数由三部分组成：Location误差，Confidence误差和分类误差。该PaddleHub Module是由800+tag,170w图片，1000w+检测框训练的大规模通用检测模型，在8个数据集上MAP平均提升5.36%，iou=0.5的准确率提升4.53%。对比于其他通用检测模型，使用该Module进行finetune，可以更快收敛，达到较优效果。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.2  

  - paddlehub >= 1.6.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)  

- ### 2、安装

  - ```shell
    $ hub install yolov3_darknet53_venus
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、API

  - ```python
    def context(trainable=True,
                pretrained=True,
                get_prediction=False)
    ```

    - 提取特征，用于迁移学习。

    - **参数**

      - trainable(bool): 参数是否可训练；<br/>
      - pretrained (bool): 是否加载预训练模型；<br/>
      - get\_prediction (bool): 是否执行预测。

    - **返回**
      - inputs (dict): 模型的输入，keys 包括 'image', 'im\_size'，相应的取值为：
        - image (Variable): 图像变量
        - im\_size (Variable): 图片的尺寸
      - outputs (dict): 模型的输出。如果 get\_prediction 为 False，输出 'head\_features'、'body\_features'，否则输出 'bbox\_out'
      - context\_prog (Program): 用于迁移学习的 Program

  - ```python
    def object_detection(paths=None,
                         images=None,
                         batch_size=1,
                         use_gpu=False,
                         score_thresh=0.5,
                         visualization=True,
                         output_dir='detection_result')
    ```

    - 预测API，检测输入图片中的所有目标的位置。

    - **参数**

      - paths (list\[str\]): 图片的路径； <br/>
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式； <br/>
      - batch\_size (int): batch 的大小；<br/>
      - use\_gpu (bool): 是否使用 GPU；<br/>
      - score\_thresh (float): 识别置信度的阈值；<br/>
      - visualization (bool): 是否将识别结果保存为图片文件；<br/>
      - output\_dir (str): 图片的保存路径，默认设为 detection\_result。

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为:
        - data (list): 检测结果，list的每一个元素为 dict，各字段为:
          - confidence (float): 识别的置信度
          - label (str): 标签
          - left (int): 边界框的左上角x坐标
          - top (int): 边界框的左上角y坐标
          - right (int): 边界框的右下角x坐标
          - bottom (int): 边界框的右下角y坐标
        - save\_path (str, optional): 识别结果的保存路径 (仅当visualization=True时存在)

  - ```python
    def save_inference_model(dirname,
                             model_filename=None,
                             params_filename=None,
                             combined=True)
    ```
    - 将模型保存到指定路径。

    - **参数**

      - dirname: 存在模型的目录名称； <br/>
      - model\_filename: 模型文件名称，默认为\_\_model\_\_； <br/>
      - params\_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)；<br/>
      - combined: 是否将参数保存到统一的一个文件中。




## 四、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install yolov3_darknet53_venus==1.0.0
    ```
