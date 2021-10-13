# inception_v4_imagenet

|模型名称|inception_v4_imagenet|
| :--- | :---: |
|类别|图像-图像分类|
|网络|Inception_V4|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|是|
|模型大小|167MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：


- ### 模型介绍

  - Inception 结构最初由 GoogLeNet 引入，因此 GoogLeNet 也被称为 Inception-v1，在 Inception-v1 的基础上引入了 Batch Normalization，得到了 Inception-v2 ；在 Inception-v2 的基础上引入了分解，得到了Inception-v3。Inception 结构有着良好的性能，且计算量低，而残差连接可加快收敛速度，可用于训练更深的网络，于是Inception V4 的作者尝试将 Inception 结构和残差连接结合，同时也设计了不用残差连接的Inception-v4。通过将三个残差和一个Inception-v4进行组合，在 ImageNet 数据集上 top-5 错误率上仅有 3.08%。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install inception_v4_imagenet
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run inception_v4_imagenet --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="inception_v4_imagenet")
    test_img_path = "/PATH/TO/IMAGE"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    ```

- ### 3、API

  - ```python
    def classification(data)
    ```

    - **参数**
      - data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率





## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install inception_v4_imagenet==1.0.0
    ```
