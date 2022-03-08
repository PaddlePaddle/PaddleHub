# U2Netp

|模型名称|U2Netp|
| :--- | :---: |
|类别|图像-图像分割|
|网络|U^2Net|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|6.7MB|
|指标|-|
|最新更新日期|2021-02-26|



## 一、模型基本信息

- ### 应用效果展示

    - 样例结果示例：
        <p align="center">
        <img src="https://ai-studio-static-online.cdn.bcebos.com/4d77bc3a05cf48bba6f67b797978f4cdf10f38288b9645d59393dd85cef58eff" width = "450" height = "300" hspace='10'/> <img src="https://ai-studio-static-online.cdn.bcebos.com/11c9eba8de6d4316b672f10b285245061821f0a744e441f3b80c223881256ca0" width = "450" height = "300" hspace='10'/>
        </p>


- ### 模型介绍

    * U2Netp的网络结构如下图，其类似于编码-解码(Encoder-Decoder)结构的 U-Net, 每个 stage 由新提出的 RSU模块(residual U-block) 组成. 例如，En_1 即为基于 RSU 构建的, 它是一个小型化的模型

    ![](https://ai-studio-static-online.cdn.bcebos.com/999d37b4ffdd49dc9e3315b7cec7b2c6918fdd57c8594ced9dded758a497913d)

    *  - 更多详情请参考：[U2Net](https://github.com/xuebinqin/U-2-Net)


## 二、安装

- ### 1、环境依赖
    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装
    - ```shell
      $ hub install U2Netp
      ```

    - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)

## 三、模型API预测
- ### 1、预测代码示例

    ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='U2Netp')

    result = model.Segmentation(
        images=[cv2.imread('/PATH/TO/IMAGE')],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir='output',
        visualization=True)
    ```
 - ### 2、API

    ```python
    def Segmentation(
            images=None,
            paths=None,
            batch_size=1,
            input_size=320,
            output_dir='output',
            visualization=False):
    ```
    - 图像前景背景分割 API

    -   **参数**
        * images (list[np.ndarray]) : 输入图像数据列表（BGR）
        * paths (list[str]) : 输入图像路径列表
        * batch_size (int) : 数据批大小
        * input_size (int) : 输入图像大小
        * output_dir (str) : 可视化图像输出目录
        * visualization (bool) : 是否可视化

    -   **返回**
        * results (list[np.ndarray]): 输出图像数据列表

## 四、更新历史

* 1.0.0

  初始发布







