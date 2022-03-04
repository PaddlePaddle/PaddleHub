# SkyAR

|Module Name|SkyAR|
| :--- | :---: |
|Category|Video editing|
|Network|UNet|
|Dataset|-|
|Fine-tuning supported or not|No|
|Module Size|206MB|
|Data indicators|-|
|Latest update date|2021-02-26|

## I. Basic Information 

- ### Application Effect Display

    - Sample results:
        * Input video:

            ![Input video](https://img-blog.csdnimg.cn/20210126142046572.gif)

        * Jupiter:

            ![Jupiter](https://img-blog.csdnimg.cn/20210125211435619.gif)
        * Rainy day:

            ![Rainy day](https://img-blog.csdnimg.cn/2021012521152492.gif)
        * Galaxy:

            ![Galaxy](https://img-blog.csdnimg.cn/20210125211523491.gif)
        * Ninth area spacecraft:

            ![Ninth area spacecraft](https://img-blog.csdnimg.cn/20210125211520955.gif)

        * Input video:

            ![Input video](https://img-blog.csdnimg.cn/20210126142038716.gif)
        * Floating castle:

            ![Floating castle](https://img-blog.csdnimg.cn/20210125211514997.gif)
        * Thunder and lightning:

            ![Thunder and lightning](https://img-blog.csdnimg.cn/20210125211433591.gif)

        * Super moon:
        
            ![Super moon](https://img-blog.csdnimg.cn/20210125211417524.gif)

- ### Module Introduction

    - SkyAR is based on [Castle in the Sky: Dynamic Sky Replacement and Harmonization in Videos](https://arxiv.org/abs/2010.11800). It mainly consists of three parts: sky matting network, motion estimation and image fusion.

    - For more information, please refer to:[SkyAR](https://github.com/jiupinjia/SkyAR)


## II. Installation

- ### 1、Environmental Dependence

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、Installation

    - ```shell
      $hub install SkyAR
      ```
    - In case of any problems during installation, please refer to:[Windows_Quickstart](../../../../docs/docs_en/get_start/windows_quickstart.md)
    | [Linux_Quickstart](../../../../docs/docs_en/get_start/linux_quickstart.md) | [Mac_Quickstart](../../../../docs/docs_en/get_start/mac_quickstart.md)  

## III. Module API Prediction

- ### 1、Prediction Code Example

    ```python
    import paddlehub as hub

    model = hub.Module(name='SkyAR')

    model.MagicSky(
        video_path=[path to input video path],
        save_path=[path to save video path]
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

    - **Parameter**

        * video_path(str)：input video path.
        * save_path(str)：save videp path.
        * config(str): SkyBox configuration, all preset configurations are as follows:  `['cloudy', 'district9ship', 'floatingcastle', 'galaxy', 'jupiter',
            'rainy', 'sunny', 'sunset', 'supermoon', 'thunderstorm'
        ]`, if you use a custom SkyBox, please set it to None.
    
        * skybox_img(str)：custom SkyBox image path
        * skybox_video(str)：custom SkyBox video path
        * is_video_sky(bool)：customize whether SkyBox is a video
        * rain_cap_path(str)：custom video path with rain
        * is_rainy(bool): whether the sky is raining
        * halo_effect(bool)：whether to open halo effect
        * auto_light_matching(bool)：whether to enable automatic brightness matching
        * relighting_factor(float): relighting factor
        * recoloring_factor(float): recoloring factor
        * skybox_center_crop(float)：skyBox center crop factor
        * preview_frames_num(int)：set the number of preview frames
        * is_show(bool)：whether to preview graphically


## IV. Release Note

- 1.0.0

  First release
