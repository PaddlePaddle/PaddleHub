## 模型概述
* SkyAR 是一种用于视频中天空置换与协调的视觉方法，该方法能够在风格可控的视频中自动生成逼真的天空背景。
* 该算法是一种完全基于视觉的解决方案，它的好处就是可以处理非静态图像，同时不受拍摄设备的限制，也不需要用户交互，可以处理在线或离线视频。
* 算法主要由三个核心组成：
  * 天空抠图网络（Sky Matting Network）：就是一种 Matting 图像分隔，用于检测视频帧中天空区域的视频，可以精确地获得天空蒙版。
  * 运动估计（Motion Estimation）：恢复天空运动的运动估计器，使生成的天空与摄像机的运动同步。
  * 图像融合（Image Blending）：将用户指定的天空模板混合到视频帧中。除此之外，还用于重置和着色，使混合结果在其颜色和动态范围内更具视觉逼真感。
* 整体框架图如下：

	![](http://p4.itc.cn/q_70/images03/20201114/42eaf00af8dd4aa4ae3c0cdc6e50b793.jpeg)
* 参考论文：Zhengxia Zou. [Castle in the Sky: Dynamic Sky Replacement and Harmonization in Videos](https://arxiv.org/abs/2010.11800). CoRR, abs/2010.118003, 2020.
* 官方开源项目： [jiupinjia/SkyAR](https://github.com/jiupinjia/SkyAR)
## 模型安装
```shell
$hub install SkyAR
```

## 效果展示
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

## API 说明

```python
def MagicSky(
        video_path, save_path, config='jupiter',
        is_rainy=False, preview_frames_num=0, is_video_sky=False, is_show=False,
        skybox_img=None, skybox_video=None, rain_cap_path=None,
        halo_effect=True, auto_light_matching=False,
        relighting_factor=0.8, recoloring_factor=0.5, skybox_center_crop=0.5
    )
```

深度估计API

**参数**

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

## 预测代码示例

```python
import paddlehub as hub

model = hub.Module(name='SkyAR')

model.MagicSky(
    video_path=[path to input video path],
    save_path=[path to save video path]
)
```

## 模型相关信息

### 模型代码

https://github.com/jm12138/SkyAR_Paddle_GUI

### 依赖

paddlepaddle >= 2.0.0rc0

paddlehub >= 2.0.0rc0
