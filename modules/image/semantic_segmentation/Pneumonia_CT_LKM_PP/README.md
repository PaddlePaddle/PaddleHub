
# Pneumonia_CT_LKM_PP

|模型名称|Pneumonia_CT_LKM_PP|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|-|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|35M|
|指标|-|
|最新更新日期|2021-02-26|


## 一、模型基本信息


- ### 模型介绍

    - 肺炎CT影像分析模型（Pneumonia-CT-LKM-PP）可以高效地完成对患者CT影像的病灶检测识别、病灶轮廓勾画，通过一定的后处理代码，可以分析输出肺部病灶的数量、体积、病灶占比等全套定量指标。值得强调的是，该系统采用的深度学习算法模型充分训练了所收集到的高分辨率和低分辨率的CT影像数据，能极好地适应不同等级CT影像设备采集的检查数据，有望为医疗资源受限和医疗水平偏低的基层医院提供有效的肺炎辅助诊断工具。

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install Pneumonia_CT_LKM_PP==1.0.0
      ```
      
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    pneumonia = hub.Module(name="Pneumonia_CT_LKM_PP")

    input_only_lesion_np_path = "/PATH/TO/ONLY_LESION_NP"
    input_both_lesion_np_path = "/PATH/TO/LESION_NP"
    input_both_lung_np_path = "/PATH/TO/LUNG_NP"

    # set input dict
    input_dict = {"image_np_path": [
                                    [input_only_lesion_np_path],
                                    [input_both_lesion_np_path, input_both_lung_np_path],
                                    ]}

    # execute predict and print the result
    results = pneumonia.segmentation(data=input_dict)
    for result in results:
        print(result)

    ```
   

- ### 2、API

    ```python
    def segmentation(data)
    ```

    - 预测API，用于肺炎CT影像分析。

    - **参数**

        * data (dict): key，str类型，"image_np_path"；value，list类型，每个元素为list类型，[用于病灶分析的影像numpy数组(文件后缀名.npy)路径, 用于肺部分割的影像numpy数组路径]，如果仅进行病灶分析不进行肺部分割，可以省略用于肺部分割的影像numpy数组路径
       

    - **返回**

        * result  (list\[dict\]): 每个元素为对应输入的预测结果。每个预测结果为dict类型：预测结果有以下字段：
            * input_lesion_np_path: 存放用于病灶分析的numpy数组路径；
            * output_lesion_np: 存放病灶分析结果，numpy数组；
            * input_lesion_np_path：存放用于肺部分割的numpy数组路径（仅当对应输入包含肺部影像numpy时存在该字段）
            * output_lung_np：存放肺部分割结果，numpy数组（仅当对应输入包含肺部影像numpy时存在该字段）


## 四、更新历史

* 1.0.0

    初始发布
