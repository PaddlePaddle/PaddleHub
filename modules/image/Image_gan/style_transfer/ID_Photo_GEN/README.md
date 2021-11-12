# ID_Photo_GEN

|模型名称|ID_Photo_GEN|
| :--- | :---: |
|类别|图像 - 图像生成|
|网络|HRNet_W18|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|28KB|
|最新更新日期|2021-02-26|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
    <p align="center">
    <img src="https://img-blog.csdnimg.cn/20201224163307901.jpg" > 
    </p>


- ### 模型介绍

  - 基于face_landmark_localization和FCN_HRNet_W18_Face_Seg模型实现的证件照生成模型，一键生成白底、红底和蓝底的人像照片


## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.0.0

- ### 2、安装

  - ```shell
    $ hub install ID_Photo_GEN
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)
 
 
## 三、模型API预测

- ### 1、预测代码示例

  - ```python
    import cv2
    import paddlehub as hub

    model = hub.Module(name='ID_Photo_GEN')

    result = model.Photo_GEN(
    images=[cv2.imread('/PATH/TO/IMAGE')],
    paths=None,
    batch_size=1,
    output_dir='output',
    visualization=True,
    use_gpu=False)
    ```

- ### 2、API

  - ```python
    def Photo_GEN(
        images=None,
        paths=None,
        batch_size=1,
        output_dir='output',
        visualization=False,
        use_gpu=False):
    ```

    - 证件照生成API

    - **参数**
        * images (list[np.ndarray]) : 输入图像数据列表（BGR）
        * paths (list[str]) : 输入图像路径列表
        * batch_size (int) : 数据批大小
        * output_dir (str) : 可视化图像输出目录
        * visualization (bool) : 是否可视化
        * use_gpu (bool) : 是否使用 GPU 进行推理

      **NOTE:** paths和images两个参数选择其一进行提供数据

    - **返回**
    
      * results (list[dict{"write":np.ndarray,"blue":np.ndarray,"red":np.ndarray}]): 输出图像数据列表


## 四、更新历史

* 1.0.0

  初始发布
