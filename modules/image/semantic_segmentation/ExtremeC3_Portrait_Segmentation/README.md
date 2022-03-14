# ExtremeC3_Portrait_Segmentation

|模型名称|ExtremeC3_Portrait_Segmentation|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|ExtremeC3|
|数据集|EG1800, Baidu fashion dataset|
|是否支持Fine-tuning|否|
|模型大小|0.038MB|
|指标|-|
|最新更新日期|2021-02-26|

## 一、模型基本信息

- ### 应用效果展示

    - 样例结果示例：
        <p align="center">
        <img src="https://ai-studio-static-online.cdn.bcebos.com/1261398a98e24184852bdaff5a4e1dbd7739430f59fb47e8b84e3a2cfb976107"  hspace='10'/> <br />
        </p>


- ### 模型介绍
    * 基于 ExtremeC3 模型实现的轻量化人像分割模型

    * 更多详情请参考： [ExtremeC3_Portrait_Segmentation](https://github.com/clovaai/ext_portrait_segmentation) 项目

## 二、安装

- ### 1、环境依赖
    - paddlepaddle >= 2.0.0  

    - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install ExtremeC3_Portrait_Segmentation
      ```
      
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测

- ### 1、预测代码示例

    ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='ExtremeC3_Portrait_Segmentation')

    result = model.Segmentation(
        images=[cv2.imread('/PATH/TO/IMAGE')],
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False)
    ```

- ### 2、API

```python
def Segmentation(
    images=None,
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=False):
```
- 人像分割 API

- **参数**
    * images (list[np.ndarray]) : 输入图像数据列表（BGR）
    * paths (list[str]) : 输入图像路径列表
    * batch_size (int) : 数据批大小
    * output_dir (str) : 可视化图像输出目录
    * visualization (bool) : 是否可视化

- **返回**
    * results (list[dict{"mask":np.ndarray,"result":np.ndarray}]): 输出图像数据列表

## 四、更新历史

* 1.0.0

  初始发布
