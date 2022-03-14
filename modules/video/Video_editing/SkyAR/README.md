# SkyAR

|模型名称|SkyAR|
| :--- | :---: |
|类别|视频-视频编辑|
|网络|UNet|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|206MB|
|指标|-|
|最新更新日期|2021-02-26|

## 一、模型基本信息

- ### 应用效果展示

    - 样例结果示例：
        * 原始视频：

            ![原始视频](https://img-blog.csdnimg.cn/20210126142046572.gif)

        * 木星：

            ![木星](https://img-blog.csdnimg.cn/20210125211435619.gif)
        * 雨天：

            ![雨天](https://img-blog.csdnimg.cn/2021012521152492.gif)
        * 银河：

            ![银河](https://img-blog.csdnimg.cn/20210125211523491.gif)
        * 第九区飞船：

            ![第九区飞船](https://img-blog.csdnimg.cn/20210125211520955.gif)
        * 原始视频：

            ![原始视频](https://img-blog.csdnimg.cn/20210126142038716.gif)
        * 漂浮城堡：

            ![漂浮城堡](https://img-blog.csdnimg.cn/20210125211514997.gif)
        * 电闪雷鸣：

            ![电闪雷鸣](https://img-blog.csdnimg.cn/20210125211433591.gif)
        * 超级月亮：

            ![超级月亮](https://img-blog.csdnimg.cn/20210125211417524.gif)

- ### 模型介绍

    - SkyAR是一种用于视频中天空置换与协调的视觉方法，主要由三个核心组成：天空抠图网络、运动估计和图像融合。

    - 更多详情请参考：[SkyAR](https://github.com/jiupinjia/SkyAR)

    - 参考论文：Zhengxia Zou. [Castle in the Sky: Dynamic Sky Replacement and Harmonization in Videos](https://arxiv.org/abs/2010.11800). CoRR, abs/2010.118003, 2020.

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装

    ```shell
    $hub install SkyAR
    ```
    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
      | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测

- ### 1、预测代码示例

    ```python
    import paddlehub as hub

    model = hub.Module(name='SkyAR')

    model.MagicSky(
        video_path="/PATH/TO/VIDEO",
        save_path="/PATH/TO/SAVE/RESULT"
    )
    ```
- ### 2、API

    ```python
    def MagicSky(
            video_path, save_path, config='jupiter',
            is_rainy=False, preview_frames_num=0, is_video_sky=False, is_show=False,
            skybox_img=None, skybox_video=None, rain_cap_path=None,
            halo_effect=True, auto_light_matching=False,
            relighting_factor=0.8, recoloring_factor=0.5, skybox_center_crop=0.5
        )
    ```

    - **参数**

        * video_path(str)：输入视频路径
        * save_path(str)：视频保存路径
        * config(str): 预设 SkyBox 配置，所有预设配置如下，如果使用自定义 SkyBox，请设置为 None：
        ```
        [
            'cloudy', 'district9ship', 'floatingcastle', 'galaxy', 'jupiter',
            'rainy', 'sunny', 'sunset', 'supermoon', 'thunderstorm'
        ]
        ```
        * skybox_img(str)：自定义的 SkyBox 图像路径
        * skybox_video(str)：自定义的 SkyBox 视频路径
        * is_video_sky(bool)：自定义 SkyBox 是否为视频
        * rain_cap_path(str)：自定义下雨效果视频路径
        * is_rainy(bool): 天空是否下雨
        * halo_effect(bool)：是否开启 halo effect
        * auto_light_matching(bool)：是否开启自动亮度匹配
        * relighting_factor(float): Relighting factor
        * recoloring_factor(float): Recoloring factor
        * skybox_center_crop(float)：SkyBox center crop factor
        * preview_frames_num(int)：设置预览帧数量，即只处理开头这几帧，设为 0，则为全部处理
        * is_show(bool)：是否图形化预览


## 四、更新历史

* 1.0.0

  初始发布
