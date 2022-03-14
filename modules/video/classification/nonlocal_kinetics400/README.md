# nonlocal_kinetics400

|模型名称|nonlocal_kinetics400|
| :--- | :---: | 
|类别|视频-视频分类|
|网络|Non-local|
|数据集|Kinetics-400|
|是否支持Fine-tuning|否|
|模型大小|129MB|
|最新更新日期|2021-02-26|
|数据指标|-|



## 一、模型基本信息

- ### 模型介绍

  - Non-local Neural Networks是由Xiaolong Wang等研究者在2017年提出的模型，主要特点是通过引入Non-local操作来描述距离较远的像素点之间的关联关系。其借助于传统计算机视觉中的non-local mean的思想，并将该思想扩展到神经网络中，通过定义输出位置和所有输入位置之间的关联函数，建立全局关联特性。Non-local模型的训练数据采用由DeepMind公布的Kinetics-400动作识别数据集。该PaddleHub Module可支持预测。


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 1.4.0
  
  - paddlehub >= 1.0.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install nonlocal_kinetics400
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)




## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    hub run nonlocal_kinetics400 --input_path "/PATH/TO/VIDEO" --use_gpu True
    ```
    
    或者
    
  - ```shell
    hub run nonlocal_kinetics400 --input_file test.txt --use_gpu True
    ```    
    
  - test.txt 存放待分类视频的存放路径；
  - Note: 该PaddleHub Module目前只支持在GPU环境下使用，在使用前，请使用下述命令指定GPU设备（设备ID请根据实际情况指定）
  
  - ```shell
    export CUDA_VISIBLE_DEVICES=0
    ```    

  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python

    import paddlehub as hub

    nonlocal = hub.Module(name="nonlocal_kinetics400")

    test_video_path = "/PATH/TO/VIDEO"

    # set input dict
    input_dict = {"image": [test_video_path]}

    # execute predict and print the result
    results = nonlocal.video_classification(data=input_dict)
    for result in results:
        print(result)
    ```
    
- ### 3、API

  - ```python
    def video_classification(data)
    ```    

    - 用于视频分类预测
    
    - **参数**

      - data(dict): dict类型，key为image，str类型；value为待分类的视频路径，list类型。


    - **返回**

      - result(list\[dict\]): list类型，每个元素为对应输入视频的预测结果。预测结果为dict类型，key为label，value为该label对应的概率值。


## 五、更新历史

* 1.0.0

  初始发布
  
  - ```shell
    $  hub install nonlocal_kinetics400==1.0.0
    ```
