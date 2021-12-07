# resnext101_32x48d_wsl

|模型名称|resnext101_32x48d_wsl|
| :--- | :---: |
|类别|图像-图像分类|
|网络|ResNeXt_wsl|
|数据集|ImageNet-2012|
|是否支持Fine-tuning|否|
|模型大小|342MB|
|最新更新日期|-|
|数据指标|-|


## 一、模型基本信息



- ### 模型介绍

  - 由于人工标注的数据集在规模上已经接近其函数极限，Facebook 的研发人员采用了一种独特的迁移学习研究，通过使用 hashtag 作为标注，在包含数十亿张社交媒体图片的数据集上进行训练，这为大规模训练转向弱监督学习(Weakly Supervised Learning) 取得了重大突破。在 ImageNet 图像识别基准上，ResNeXt101_32x48d_wsl 的 Top-1 达到了 85.4% 的准确率。该 PaddleHub Module结构为 ResNeXt101_32x48d_wsl，接受输入图片大小为 224 x 224 x 3，支持直接通过命令行或者 Python 接口进行预测。

## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.6.0  

  - paddlehub >= 1.0.0  | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)


- ### 2、安装

  - ```shell
    $ hub install resnext101_32x48d_wsl
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run resnext101_32x48d_wsl --input_path "/PATH/TO/IMAGE"
    ```
  - 通过命令行方式实现图像分类模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    classifier = hub.Module(name="resnext101_32x48d_wsl")
    test_img_path = "/PATH/TO/IMAGE"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    ```

- ### 3、API

  - ```python
    def classification(data)
    ```
    - 分类接口API。
    - **参数**
      - data：dict类型，key为image，str类型，value为待检测的图片路径，list类型。

    - **返回**
      - result：list类型，每个元素为对应输入图片的预测结果。预测结果为dict类型，key为该图片分类结果label，value为该label对应的概率





## 四、更新历史

* 1.0.0

  初始发布

  - ```shell
    $ hub install resnext101_32x48d_wsl==1.0.0
    ```
