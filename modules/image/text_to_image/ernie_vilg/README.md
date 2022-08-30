  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/187387422-f6c9ccab-7fda-416e-a24d-7d6084c46f67.jpg"  width = "80%" hspace='10'/>

# PaddleHub ERNIE-ViLG

|模型名称|ernie_vilg|
| :--- | :---: |
|类别|图像-文图生成|
|网络|ERNIE-ViLG|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-08-02|
|数据指标|-|

## 一、模型基本信息

### 应用效果展示

  - 输入文本 "戴眼镜的猫"  风格 "油画"

  - 输出图像
  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/187395109-8ab830bb-4559-41a2-97a1-0a45d217abd6.png"  width = "80%" hspace='10'/>
  <br />


### 模型介绍

文心ERNIE-ViLG参数规模达到100亿，是目前为止全球最大规模中文跨模态生成模型，在文本生成图像、图像描述等跨模态生成任务上效果全球领先，在图文生成领域MS-COCO、COCO-CN、AIC-ICC等数据集上取得最好效果。你可以输入一段文本描述以及生成风格，模型就会根据输入的内容自动创作出符合要求的图像。

## 二、安装

- ### 1、环境依赖

  - paddlepaddle >= 2.0.0

  - paddlehub >= 2.2.0    | [如何安装PaddleHub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install ernie_vilg
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)


## 三、模型API预测  

- ### 1、命令行预测

  - ```shell
    $ hub run ernie_vilg --text_prompts "宁静的小镇" --style "油画" --output_dir ernie_vilg_out
    ```

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub

    module = hub.Module(name="ernie_vilg")
    text_prompts = ["宁静的小镇"]
    images = module.generate_image(text_prompts=text_prompts, style='油画', output_dir='./ernie_vilg_out/')  
    ```

- ### 3、API

  - ```python
    def generate_image(
              text_prompts:str,
              style: Optional[str] = "油画",
              topk: Optional[int] = 6,
              output_dir: Optional[str] = 'ernievilg_output')
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。
      - style(Optional[str]): 生成图像的风格，当前支持'油画','水彩','粉笔画','卡通','儿童画','蜡笔画','探索无限'。
      - topk(Optional[int]): 保存前多少张图，最多保存6张。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"ernievilg_output"。


    - **返回**
      - images(List(PIL.Image)): 返回生成的所有图像列表，PIL的Image格式。

## 四、服务部署

- PaddleHub Serving可以部署一个在线文图生成服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m ernie_vilg
    ```

  - 这样就完成了一个文图生成的在线服务API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果。

  - ```python
    import requests
    import json
    import cv2
    import base64
    from io import BytesIO
    from PIL import Image

    # 发送HTTP请求
    data = {'text_prompts': '巨大的白色城堡'}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/ernie_vilg"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 获取返回结果
    for i, result in enumerate(r.json()["results"]):
      image = Image.open(BytesIO(base64.b64decode(result)))
      image.save('result_{}.png'.format(i))


## 五、更新历史

* 1.0.0

  初始发布

  ```shell
  $ hub install ernie_vilg == 1.0.0
  ```




## 六、 Prompt 指南



这是一份如何调整 Prompt 得到更漂亮的图片的经验性文档。我们的结果和经验都来源于[文心 ERNIE-ViLG Demo](https://wenxin.baidu.com/moduleApi/ernieVilg) 和[社区的资料](#related-work)。

什么是 Prompt？Prompt 是输入到 Demo 中的文字，可以是一个实体，例如猫；也可以是一串富含想象力的文字，例如：『夕阳日落时，天边有巨大的云朵，海面波涛汹涌，风景，胶片感』。不同的 Prompt 对于生成的图像质量影响非常大。所以也就有了下面所有的 Prompt 的一些经验性技巧。

| ![174_蒙娜丽莎，赛博朋克，宝丽来，33毫米,蒸汽波艺术_000-1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/174_蒙娜丽莎，赛博朋克，宝丽来，33毫米,蒸汽波艺术_000-1.jpg) |
| :----------------------------------------------------------: |
|        蒙娜丽莎，赛博朋克，宝丽来，33毫米,蒸汽波艺术         |




## 前言

Prompt 的重要性如此重要，以至于我们需要构造一个示例来进行一次说明。

如下图，[文心 ERNIE-ViLG Demo](https://wenxin.baidu.com/moduleApi/ernieVilg) 中，『卡通』模式下，输入的 Prompt 为『橘猫』，以及 『卡通』模型式下『极乐迪斯科里的猫, 故障艺术』两个示例，能够看出来后者的细节更多，呈现的图片也更加的风格化。

开放风格限制（本质上就是在 Prompt 中不加入风格控制词），即下图图3，得到的图片细节更多、也更加真实，同时还保留了比较强烈的风格元素。所以后面的所有内容，都将围绕着如何构造更好的 Prompt 进行资料的整理。

| ![橘猫](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/极乐猫0.jpg) | ![极乐迪斯科里的猫](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/极乐猫1.jpg) | ![极乐迪斯科里的猫](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/极乐猫3.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|                         “橘猫”(卡通)                         |              “极乐迪斯科里的猫, 故障艺术”(卡通)              | “极乐迪斯科里的猫, 故障艺术” (探索无限) |

| ![cat-hd](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/cat-hd.jpg) |
| :----------------------------: |
|   极乐迪斯科里的猫,故障艺术    |



## 呼吁与准则

机器生成图片的最终目的还是便捷地为人类创造美的作品。而技术不是十全十美的，不能保证每次生成的图像都能够尽善尽美。因此呼吁所有相关玩家，如果想分享作品，那就分享那些美感爆棚的作品！

算法生成的图片难免会受到数据的影响，从而导致生成的图片是有数据偏见的。因此在分享机器生成图片到社交媒体之前，请三思当前的图片是不是含有：令人不适的、暴力的、色情的内容。如果有以上的内容请自行承担法律后果。


<span id = "p-design">   </span>
## Prompt 的设计

如何设计 Prompt，下文大概会通过4个方面来说明：[Prompt 公式](#p-eq)，[Prompt 原则](#p-principle)，[Prompt 主体](#p-entity)、[Prompt 修饰词](#p-modifier)。

需要注意的是，这里的 Prompt 公式仅仅是个入门级别的参考，是经验的简单总结，在熟悉了 Prompt 的原理之后，可以尽情的发挥脑洞修改 Prompt。





<span id = "p-eq">   </span>
## Prompt 公式

$$
Prompt = [形容词] [主语] ，[细节设定]， [修饰语或者艺术家]
$$

按照这个公式，我们首先构造一个形容词加主语的案例。 这里我构造的是 戴着眼镜的猫， 风格我选择的是油画风格，然后我再添加一些细节设定，这里我给的是 漂浮在宇宙中， 可以看到 ，猫猫的后面出现了很多天体。

| ![猫1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/猫1.jpg) | ![猫2](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/猫2.jpg) | ![猫3](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/猫3.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                     “戴着眼镜的猫”(油画)                     |              “戴着眼镜的猫，漂浮在宇宙中”(油画)              |         “戴着眼镜的猫，漂浮在宇宙中，高更风格”(油画)         |

最后我们想让我们的照片风格更加有艺术性的效果， 我们选择的艺术家是高更， 可以看到图像的画风有了更强的艺术风格。



<span id = "p-principle">   </span>
## Prompt 设计原则

### Prompt 简单原则: 清楚地陈述

除了公式之外，也有一些简单的 Prompt设计原则分享给大家：即**清楚的陈述**。

例如我们如果是简单的输入风景的话，往往模型不知道我们想要的风景是什么样子的(下图1)。我们要去尽量的幻想风景的样子，然后变成语言描述。 例如我想像的是日落时，海边的风景， 那我就构造了 Prompt 『夕阳日落时，阳光落在云层上，海面波光粼粼，风景』(下图2)。 进一步的，我想风格化我的图像，所以我在结尾的部分，增加了『胶片感』来让图片的色彩更加好看一些(下图3)。但是云彩的细节丢失了一些，进一步的我再增加天边巨大云朵这一个细节，让我的图片朝着我想要的样子靠的更进一步(下图4)。

| ![猫1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/风景1.jpg) |            ![猫2](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/风景2.jpg)            |                ![猫3](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/风景3.jpg)                | ![猫3](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/风景4.jpg)                               |
| :------------------------: | :----------------------------------------------: | :------------------------------------------------------: | -------------------------------------------------------- |
|           “风景”           | “夕阳日落时，阳光落在云层上，海面波光粼粼，风景” | “夕阳日落时，阳光落在云层上，海面波涛汹涌，风景，胶片感” | 夕阳日落时，天边有巨大的云朵，海面波涛汹涌，风景，胶片感 |



<span id = "p-entity">   </span>
## Prompt 主体的选择

Prompt 的主体可以是千奇百怪、各种各样的。这里我挑了几个简单的容易出效果的主体示例和一些能够营造特殊氛围的氛围词来激发大家的灵感。



| ![宇航员](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/宇航员.jpg) | ![孤岛](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/孤岛.jpg) | ![白色城堡](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/白色城堡.jpg) | ![机器人](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/机器人.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                            宇航员                            |                             孤岛                             |                           白色城堡                           |                            机器人                            |
| ![巫师](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/巫师.jpg) | ![罗马城](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/罗马城.jpg) | ![海鸥](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/海鸥.jpg) | ![气球](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/气球.jpg) |
|                             巫师                             |                            罗马城                            |                             海鸥                             |                             气球                             |





| ![霓虹灯](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/霓虹灯.jpg) | ![烟](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/烟.jpg) | ![漩涡](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/漩涡.jpg) |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                      …日落，霓虹灯…薄雾                      |                             …烟…                             |                    …燃烧漩涡, …烟雾和碎片                    |
| ![废墟](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/废墟.jpg) | ![光之](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/光之.jpg) | ![巨大的](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/巨大的.jpg) |
|                            …废墟…                            |                            光之…                             |                           巨大的…                            |



<span id = "p-modifier">   </span>
## Prompt 修饰词

如果想让生成的图片更加的艺术化、风格话，可以考虑在 Prompt 中添加艺术修饰词。艺术修饰词可以是一些美术风格(例如表现主义、抽象主义等)，也可以是一些美学词汇（蒸汽波艺术、故障艺术等），也可以是一些摄影术语（80mm摄像头、浅景深等），也可以是一些绘图软件（虚幻引擎、C4D等）。

按照这样的规律，我们在两个输入基准上 ：

> 一只猫坐在椅子上，戴着一副墨镜
>
> 日落时的城市天际线
>

通过构造『输入 + Prompt 修饰词』来展示不同修饰词的效果 (这里的策略参考了[资料](https://docs.google.com/document/d/11WlzjBT0xRpQhP9tFMtxzd0q6ANIdHPUBkMV-YB043U/edit))。

需要注意的是，不是所有的 Prompt 对于所有的修饰词都会发生反应。所以查阅 Prompt 修饰词的过程中，会发现部分的 Prompt 修饰词只能对两个基准中的一个生效。这是很正常的，因为 Prompt 的调优是一个反复的试错的过程。接下来，大家结合如下的 Prompt 修饰词， Happy Prompting 吧！



### 复古未来主义风格

| ![00472_000_一只猫坐在椅子上，戴着一副墨镜,复古未来主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00472_000_一只猫坐在椅子上，戴着一副墨镜,复古未来主义风格.jpg) | ![00472_000_日落时的城市天际线,复古未来主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00472_000_日落时的城市天际线,复古未来主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,复古未来主义风格              | 日落时的城市天际线,复古未来主义风格                          |



### 粉彩朋克风格

| ![00017_004_一只猫坐在椅子上，戴着一副墨镜，粉彩朋克风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00017_004_一只猫坐在椅子上，戴着一副墨镜，粉彩朋克风格.jpg) | ![00029_001_日落时的城市天际线，粉彩朋克风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00029_001_日落时的城市天际线，粉彩朋克风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,粉彩朋克风格                  | 日落时的城市天际线,粉彩朋克风格                              |

### 史前遗迹风格

| ![00443_005_一只猫坐在椅子上，戴着一副墨镜,史前遗迹风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00443_005_一只猫坐在椅子上，戴着一副墨镜,史前遗迹风格.jpg) | ![00443_005_日落时的城市天际线,史前遗迹风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00443_005_日落时的城市天际线,史前遗迹风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,史前遗迹风格                  | 日落时的城市天际线,史前遗迹风格                              |




### 波普艺术风格

| ![00434_005_一只猫坐在椅子上，戴着一副墨镜,波普艺术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00434_005_一只猫坐在椅子上，戴着一副墨镜,波普艺术风格.jpg) | ![00434_002_日落时的城市天际线,波普艺术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00434_002_日落时的城市天际线,波普艺术风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,波普艺术风格                  | 日落时的城市天际线,后世界末日风格                            |



### 迷幻风格

| ![00451_000_一只猫坐在椅子上，戴着一副墨镜,迷幻药风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00451_000_一只猫坐在椅子上，戴着一副墨镜,迷幻药风格.jpg) | ![00451_001_日落时的城市天际线,迷幻药风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00451_001_日落时的城市天际线,迷幻药风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,迷幻风格                      | 日落时的城市天际线,迷幻风格                                  |


### 赛博朋克风格

| ![00142_003_一只猫坐在椅子上，戴着一副墨镜,赛博朋克风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00142_003_一只猫坐在椅子上，戴着一副墨镜,赛博朋克风格.jpg) | ![00142_000_日落时的城市天际线,赛博朋克风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00142_000_日落时的城市天际线,赛博朋克风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,赛博朋克风格                  | 日落时的城市天际线,赛博朋克风格                              |


### 纸箱风格


| ![00081_000_一只猫坐在椅子上，戴着一副墨镜,纸箱风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00081_000_一只猫坐在椅子上，戴着一副墨镜,纸箱风格.jpg) | ![00081_000_日落时的城市天际线,纸箱风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00081_000_日落时的城市天际线,纸箱风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,纸箱风格                      | 日落时的城市天际线,纸箱风格                                  |

### 未来主义风格

| ![00083_000_一只猫坐在椅子上，戴着一副墨镜,未来主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00083_000_一只猫坐在椅子上，戴着一副墨镜,未来主义风格.jpg) | ![00083_002_日落时的城市天际线,未来主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00083_002_日落时的城市天际线,未来主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,未来主义风格                  | 一只猫坐在椅子上，戴着一副墨镜,未来主义风格                  |



###  抽象技术风格

| ![00000_003_一只猫坐在椅子上，戴着一副墨镜, 抽象技术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00000_003_一只猫坐在椅子上，戴着一副墨镜,抽象技术风格.jpg) | ![00000_004_日落时的城市天际线,抽象技术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00000_004_日落时的城市天际线,抽象技术风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,抽象技术风格                  | 日落时的城市天际线,抽象技术风格                              |




### 海滩兔风格


| ![00049_001_一只猫坐在椅子上，戴着一副墨镜,海滩兔风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00049_001_一只猫坐在椅子上，戴着一副墨镜,海滩兔风格.jpg) | ![00049_003_日落时的城市天际线,海滩兔风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00049_003_日落时的城市天际线,海滩兔风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,海滩兔风格                    | 日落时的城市天际线,海滩兔风格                                |


### 粉红公主风格

| ![00038_004_一只猫坐在椅子上，戴着一副墨镜，粉红公主风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00038_004_一只猫坐在椅子上，戴着一副墨镜，粉红公主风格.jpg) | ![00046_004_日落时的城市天际线，粉红公主风格-1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00046_004_日落时的城市天际线，粉红公主风格-1.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,粉红公主风格                  | 日落时的城市天际线,粉红公主风格                              |


### 嬉皮士风格

| ![00275_002_一只猫坐在椅子上，戴着一副墨镜,嬉皮士风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00275_002_一只猫坐在椅子上，戴着一副墨镜,嬉皮士风格.jpg) | ![00275_001_日落时的城市天际线,嬉皮士风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00275_001_日落时的城市天际线,嬉皮士风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,嬉皮士风格                    | 日落时的城市天际线,嬉皮士风格                                |

### 幻象之城风格

| ![00288_000_一只猫坐在椅子上，戴着一副墨镜,幻象之城风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00288_000_一只猫坐在椅子上，戴着一副墨镜,幻象之城风格.jpg) | ![00288_004_日落时的城市天际线,幻象之城风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00288_004_日落时的城市天际线,幻象之城风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,幻象之城风格                  | 日落时的城市天际线,幻象之城风格                              |


### 美人鱼风格

| ![00351_002_一只猫坐在椅子上，戴着一副墨镜,美人鱼风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00351_002_一只猫坐在椅子上，戴着一副墨镜,美人鱼风格.jpg) | ![00351_000_日落时的城市天际线,美人鱼风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00351_000_日落时的城市天际线,美人鱼风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,美人鱼风格                    | 日落时的城市天际线,美人鱼风格                                |


### 迷宫物语风格


| ![00382_005_一只猫坐在椅子上，戴着一副墨镜,迷宫物语风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00382_005_一只猫坐在椅子上，戴着一副墨镜,迷宫物语风格.jpg) | ![00382_000_日落时的城市天际线,迷宫物语风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00382_000_日落时的城市天际线,迷宫物语风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,迷宫物语风格                  | 日落时的城市天际线,迷宫物语风格                              |

### 仙女风格


| ![00397_003_一只猫坐在椅子上，戴着一副墨镜,仙女风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00397_003_一只猫坐在椅子上，戴着一副墨镜,仙女风格.jpg) | ![00397_004_日落时的城市天际线,仙女风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00397_004_日落时的城市天际线,仙女风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,仙女风格                      | 日落时的城市天际线,仙女风格                                  |





### Low Poly 风格

| ![猫low-poly风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/猫low-poly风格.jpg) | ![sky-line-low-poly](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/sky-line-low-poly.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, low poly 风格                | 日落时的城市天际线, low-poly                                 |




### 浮世绘风格

| ![00564_001_一只猫坐在椅子上，戴着一副墨镜,浮世绘风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00564_001_一只猫坐在椅子上，戴着一副墨镜,浮世绘风格.jpg) | ![00564_002_日落时的城市天际线,浮世绘风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00564_002_日落时的城市天际线,浮世绘风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,浮世绘风格                    | 日落时的城市天际线,浮世绘风格                                |

### 矢量心风格

| ![00573_001_一只猫坐在椅子上，戴着一副墨镜,矢量心风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00573_001_一只猫坐在椅子上，戴着一副墨镜,矢量心风格.jpg) | ![00573_005_日落时的城市天际线,矢量心风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00573_005_日落时的城市天际线,矢量心风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,矢量心风格                    | 日落时的城市天际线,矢量心风格                                |


### 摩托车手风格


| ![00051_000_一只猫坐在椅子上，戴着一副墨镜,摩托车手风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00051_000_一只猫坐在椅子上，戴着一副墨镜,摩托车手风格.jpg) | ![日落时的城市天际线,摩托车手风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/日落时的城市天际线,摩托车手风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,摩托车手风格                  | 日落时的城市天际线,摩托车手风格                              |



### 孟菲斯公司风格


| ![00114_001_一只猫坐在椅子上，戴着一副墨镜,孟菲斯公司风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00114_001_一只猫坐在椅子上，戴着一副墨镜,孟菲斯公司风格.jpg) | ![00114_002_日落时的城市天际线,孟菲斯公司风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00114_002_日落时的城市天际线,孟菲斯公司风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,孟菲斯公司风格                | 日落时的城市天际线,孟菲斯公司风格                            |


### 泥塑风格


| ![一只猫坐在椅子上，戴着一副墨镜, 泥塑风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/一只猫坐在椅子上戴着一副墨镜泥塑风格.jpg) | ![00013_002_日落时的城市天际线, 泥塑](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00013_002_日落时的城市天际线,泥塑.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, 泥塑风格                     | 日落时的城市天际线, 泥塑风格                                 |




### 苔藓风格

| ![00006_001_一只猫坐在椅子上，戴着一副墨镜，苔藓风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00006_001_一只猫坐在椅子上，戴着一副墨镜，苔藓风格.jpg) | ![00004_004_日落时的城市天际线，苔藓风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00004_004_日落时的城市天际线，苔藓风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,苔藓风格                      | 日落时的城市天际线,苔藓风格                                  |



### 新浪潮风格

| ![00389_000_一只猫坐在椅子上，戴着一副墨镜,新浪潮风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00389_000_一只猫坐在椅子上，戴着一副墨镜,新浪潮风格.jpg) | ![00389_005_日落时的城市天际线,新浪潮风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00389_005_日落时的城市天际线,新浪潮风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,新浪潮风格                    | 日落时的城市天际线,新浪潮风格                                |

### 嘻哈风格

| ![00274_000_一只猫坐在椅子上，戴着一副墨镜,嘻哈风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00274_000_一只猫坐在椅子上，戴着一副墨镜,嘻哈风格.jpg) | ![00274_005_日落时的城市天际线,嘻哈风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00274_005_日落时的城市天际线,嘻哈风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,嘻哈风格                      | 日落时的城市天际线,嘻哈风格                                  |

### 矢量图

| ![00177_001_一只猫坐在椅子上，戴着一副墨镜, 矢量图](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00177_001_一只猫坐在椅子上戴着一副墨镜矢量图.jpg) | ![00020_002_日落时的城市天际线, 矢量图](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00020_002_日落时的城市天际线矢量图.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, 矢量图                       | 日落时的城市天际线, 矢量图                                   |

### 铅笔艺术


| ![00203_000_一只猫坐在椅子上，戴着一副墨镜, 铅笔艺术](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00203_000_一只猫坐在椅子上戴着一副墨镜铅笔艺术.jpg) | ![00053_000_日落时的城市天际线, 铅笔艺术](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00053_000_日落时的城市天际线铅笔艺术.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, 铅笔艺术                     | 日落时的城市天际线, 铅笔艺术                                 |


###  女巫店风格

| ![00606_001_一只猫坐在椅子上，戴着一副墨镜,女巫店风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00606_001_一只猫坐在椅子上，戴着一副墨镜,女巫店风格.jpg) | ![00606_000_日落时的城市天际线,女巫店风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00606_000_日落时的城市天际线,女巫店风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,女巫店风格                    | 日落时的城市天际线,女巫店风格                                |



### 4D 建模


| ![00230_000_一只猫坐在椅子上，戴着一副墨镜, 4D 建模](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00230_000_一只猫坐在椅子上戴着一副墨镜4D建模.jpg) | ![00082_001_日落时的城市天际线, 4D 建模](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00082_001_日落时的城市天际线4D建模.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, 4D 建模                      | 日落时的城市天际线, 4D 建模                                  |



### 水彩墨风格


| ![00280_004_一只猫坐在椅子上，戴着一副墨镜, 水彩墨风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00280_004_一只猫坐在椅子上，戴着一副墨镜,水彩墨风格.jpg) | ![00130_004_日落时的城市天际线, 水彩墨风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00130_004_日落时的城市天际线,水彩墨风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜, 水彩墨风格                   | 日落时的城市天际线, 水彩墨风格                               |



###  酸性精灵风格

| ![00001_004_一只猫坐在椅子上，戴着一副墨镜,酸性精灵风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00001_004_一只猫坐在椅子上，戴着一副墨镜,酸性精灵风格.jpg) | ![00001_004_日落时的城市天际线,酸性精灵风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00001_004_日落时的城市天际线,酸性精灵风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,酸性精灵风格                  | 日落时的城市天际线,酸性精灵风格                              |


### 海盗风格

| ![00427_002_一只猫坐在椅子上，戴着一副墨镜,海盗风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00427_002_一只猫坐在椅子上，戴着一副墨镜,海盗风格.jpg) | ![00427_000_日落时的城市天际线,海盗风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00427_000_日落时的城市天际线,海盗风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 日落时的城市天际线,海盗风格                                  | 一只猫坐在椅子上，戴着一副墨镜,海盗风格                      |



### 古埃及风格


| ![00017_005_一只猫坐在椅子上，戴着一副墨镜,古埃及风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00017_005_一只猫坐在椅子上，戴着一副墨镜,古埃及风格.jpg) | ![00017_003_日落时的城市天际线,古埃及风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00017_003_日落时的城市天际线,古埃及风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,古埃及风格                    | 日落时的城市天际线,古埃及风格                                |

### 风帽风格


| ![戴着帽子的猫](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/戴着帽子的猫.jpg) | ![戴着帽子的城市](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/戴着帽子的城市.jpg) |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,风帽风格                   | 日落时的城市天际线,风帽风格                                  |

### 装饰艺术风格


| ![00029_000_一只猫坐在椅子上，戴着一副墨镜,装饰艺术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00029_000_一只猫坐在椅子上，戴着一副墨镜,装饰艺术风格.jpg) | ![00029_005_日落时的城市天际线,装饰艺术风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00029_005_日落时的城市天际线,装饰艺术风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,装饰艺术风格                  | 日落时的城市天际线,装饰艺术风格                              |

### 极光风格


| ![00035_004_一只猫坐在椅子上，戴着一副墨镜,极光风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00035_004_一只猫坐在椅子上，戴着一副墨镜,极光风格.jpg) | ![00035_003_日落时的城市天际线,极光风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00035_003_日落时的城市天际线,极光风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,极光风格                      | 日落时的城市天际线,极光风格                                  |

###  秋天风格


| ![00036_005_一只猫坐在椅子上，戴着一副墨镜,秋天风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00036_005_一只猫坐在椅子上，戴着一副墨镜,秋天风格.jpg) | ![00036_003_日落时的城市天际线,秋天风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00036_003_日落时的城市天际线,秋天风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 日落时的城市天际线,秋天风格                                  | 一只猫坐在椅子上，戴着一副墨镜,秋天风格                      |

### 巴洛克风格


| ![00046_002_一只猫坐在椅子上，戴着一副墨镜,巴洛克风格风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00046_002_一只猫坐在椅子上，戴着一副墨镜,巴洛克风格风格.jpg) | ![00046_003_日落时的城市天际线,巴洛克风格风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00046_003_日落时的城市天际线,巴洛克风格风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,巴洛克风格                    | 日落时的城市天际线,巴洛克风格                                |

### 立体主义风格

| ![00128_002_一只猫坐在椅子上，戴着一副墨镜,立体主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00128_002_一只猫坐在椅子上，戴着一副墨镜,立体主义风格.jpg) | ![00128_004_日落时的城市天际线,立体主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00128_004_日落时的城市天际线,立体主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,立体主义风格                  | 日落时的城市天际线,立体主义风格                              |


### 黑暗自然主义风格

| ![00147_002_一只猫坐在椅子上，戴着一副墨镜,黑暗自然主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00147_002_一只猫坐在椅子上，戴着一副墨镜,黑暗自然主义风格.jpg) | ![00147_004_日落时的城市天际线,黑暗自然主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00147_004_日落时的城市天际线,黑暗自然主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,黑暗自然主义风格              | 日落时的城市天际线,黑暗自然主义风格                          |

### 表现主义风格

| ![00190_001_一只猫坐在椅子上，戴着一副墨镜,表现主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00190_001_一只猫坐在椅子上，戴着一副墨镜,表现主义风格.jpg) | ![00190_000_日落时的城市天际线,表现主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00190_000_日落时的城市天际线,表现主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,表现主义风格                  | 日落时的城市天际线,表现主义风格                              |

### 野兽派风格

| ![00200_000_一只猫坐在椅子上，戴着一副墨镜,野兽派风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00200_000_一只猫坐在椅子上，戴着一副墨镜,野兽派风格.jpg) | ![00200_002_日落时的城市天际线,野兽派风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00200_002_日落时的城市天际线,野兽派风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,野兽派风格                    | 日落时的城市天际线,野兽派风格                                |

### 鬼魂风格

| ![00226_001_一只猫坐在椅子上，戴着一副墨镜,鬼魂风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00226_001_一只猫坐在椅子上，戴着一副墨镜,鬼魂风格.jpg) | ![00226_002_日落时的城市天际线,鬼魂风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00226_002_日落时的城市天际线,鬼魂风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,鬼魂风格                      | 日落时的城市天际线,鬼魂风格                                  |

### 印象主义风格

| ![00289_000_一只猫坐在椅子上，戴着一副墨镜,印象主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00289_000_一只猫坐在椅子上，戴着一副墨镜,印象主义风格.jpg) | ![00289_001_日落时的城市天际线,印象主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00289_001_日落时的城市天际线,印象主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,印象主义风格                  | 日落时的城市天际线,印象主义风格                              |

### 卡瓦伊风格

| ![00305_001_一只猫坐在椅子上，戴着一副墨镜,卡瓦伊风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00305_001_一只猫坐在椅子上，戴着一副墨镜,卡瓦伊风格.jpg) | ![00305_000_日落时的城市天际线,卡瓦伊风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00305_000_日落时的城市天际线,卡瓦伊风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,卡瓦伊风格                    | 日落时的城市天际线,卡瓦伊风格                                |

### 极简主义风格

| ![00362_004_一只猫坐在椅子上，戴着一副墨镜,极简主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00362_004_一只猫坐在椅子上，戴着一副墨镜,极简主义风格.jpg) | ![00362_002_日落时的城市天际线,极简主义风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00362_002_日落时的城市天际线,极简主义风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,极简主义风格                  | 日落时的城市天际线,极简主义风格                              |

### 水井惠郎风格

| ![00364_000_一只猫坐在椅子上，戴着一副墨镜,水井惠郎风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00364_000_一只猫坐在椅子上，戴着一副墨镜,水井惠郎风格.jpg) | ![00364_000_日落时的城市天际线,水井惠郎风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00364_000_日落时的城市天际线,水井惠郎风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,水井惠郎风格                  | 日落时的城市天际线,水井惠郎风格                              |

###  照片写实风格

| ![00423_000_一只猫坐在椅子上，戴着一副墨镜,照片写实风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00423_000_一只猫坐在椅子上，戴着一副墨镜,照片写实风格.jpg) | ![00423_002_日落时的城市天际线,照片写实风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00423_002_日落时的城市天际线,照片写实风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,照片写实风格                  | 日落时的城市天际线,照片写实风格                              |


### 像素可爱风格

| ![00428_005_一只猫坐在椅子上，戴着一副墨镜,像素可爱风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00428_005_一只猫坐在椅子上，戴着一副墨镜,像素可爱风格.jpg) | ![00428_005_日落时的城市天际线,像素可爱风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00428_005_日落时的城市天际线,像素可爱风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,像素可爱风格                  | 日落时的城市天际线,像素可爱风格                              |



### 雨天风格

| ![00067_002_一只猫坐在椅子上，戴着一副墨镜，雨天风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00067_002_一只猫坐在椅子上，戴着一副墨镜，雨天风格.jpg) | ![00050_003_日落时的城市天际线，雨天风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00050_003_日落时的城市天际线，雨天风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 日落时的城市天际线,雨天风格                                  | 一只猫坐在椅子上，戴着一副墨镜,雨天风格                      |

### 湿漉漉的风格

| ![00523_005_一只猫坐在椅子上，戴着一副墨镜,湿漉漉的风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00523_005_一只猫坐在椅子上，戴着一副墨镜,湿漉漉的风格.jpg) | ![00523_001_日落时的城市天际线,湿漉漉的风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00523_001_日落时的城市天际线,湿漉漉的风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,湿漉漉的风格                  | 日落时的城市天际线,湿漉漉的风格                              |


### 维京人风格

| ![00577_004_一只猫坐在椅子上，戴着一副墨镜,维京人风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00577_004_一只猫坐在椅子上，戴着一副墨镜,维京人风格.jpg) | ![00577_005_日落时的城市天际线,维京人风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00577_005_日落时的城市天际线,维京人风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,维京人风格                    | 日落时的城市天际线,维京人风格                                |

### 后印象主义


| ![一只猫坐在椅子上，戴着一副墨镜,风格：后印象主义](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style/一只猫坐在椅子上，戴着一副墨镜,风格：后印象主义.jpg) | ![日落时的城市天际线, 风格：后印象主义-v2](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style/日落时的城市天际线,风格：后印象主义-v2.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,风格：后印象主义              | 日落时的城市天际线, 风格：后印象主义-v2                      |

### 素人主义


| ![一只猫坐在椅子上，戴着一副墨镜,风格：素人主义](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style/一只猫坐在椅子上，戴着一副墨镜,风格：素人主义.jpg) | ![日落时的城市天际线,风格：素人艺术](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style/日落时的城市天际线,风格：素人艺术.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,风格：素人主义                | 日落时的城市天际线, 风格：素人艺术                           |



### 碎核风格


| ![00064_000_一只猫坐在椅子上，戴着一副墨镜,碎核风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00064_000_一只猫坐在椅子上，戴着一副墨镜,碎核风格.jpg) | ![00064_002_日落时的城市天际线,碎核风格](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/art-style-1024/00064_002_日落时的城市天际线,碎核风格.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只猫坐在椅子上，戴着一副墨镜,碎核风格                      | 日落时的城市天际线,碎核风格                                  |







## Prompt 更多信息

### 概念组合

![赛博朋克中国山水园林](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/赛博朋克中国山水园林.jpg)

## ShowCase

更多 ShowCase 和创意 Prompt，可以参考我的[社交账号](#关注我) 或者是 http://youpromptme.cn/#/gallery/ (建设中)

### 故障艺术

| ![076_时钟故障，时间故障，概念艺术，艺术站总部，pixiv趋势，cgsociety,蒸汽波艺术_004-1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/076_时钟故障，时间故障，概念艺术，艺术站总部，pixiv趋势，cgsociety,蒸汽波艺术_004-1.jpg) | ![024_巨大的纯白色城堡-油画,故障艺术_005-1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/024_巨大的纯白色城堡-油画,故障艺术_005-1.jpg) | ![065_Yggdrasil，世界树和地球融合在一起,故障艺术_009](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/065_Yggdrasil，世界树和地球融合在一起,故障艺术_009.jpg) | ![106_在百货公司和工厂的高商业需求中，未来复古科幻幻想对象或设备的专业概念艺术,故障艺术_005](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/106_在百货公司和工厂的高商业需求中，未来复古科幻幻想对象或设备的专业概念艺术,故障艺术_005.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| _时钟故障，时间故障，概念艺术，艺术站总部，pixiv趋势，cgsociety,蒸汽波艺术 | 巨大的纯白色城堡-油画,故障艺术                               | Yggdrasil，世界树和地球融合在一起,故障艺术                   | 在百货公司和工厂的高商业需求中，未来复古科幻幻想对象或设备的专业概念艺术,故障艺术 |



### 蒸汽波艺术

| ![185_荒岛,蒸汽波艺术_000-1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/185_荒岛,蒸汽波艺术_000-1.jpg) | ![060_Christoph-Vacher和Kevin-sloan创作的广阔幻想景观,蒸汽波艺术_007](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/060_Christoph-Vacher和Kevin-sloan创作的广阔幻想景观,蒸汽波艺术_007.jpg) | ![戴着眼镜的猫，蒸汽波艺术, vaporwave art 02](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/戴着眼镜的猫，蒸汽波艺术,vaporwaveart02.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 荒岛,蒸汽波艺术                                              | Christoph-Vacher和Kevin-sloan创作的广阔幻想景观,蒸汽波艺术   | 戴着眼镜的猫，蒸汽波艺术                                     |


### 包豪斯艺术

| ![007_一只海鸥和史蒂文·西格正在进行一场凝视比赛，绘画,包豪斯_002](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/007_一只海鸥和史蒂文·西格正在进行一场凝视比赛，绘画,包豪斯_002.jpg) | ![033_梵高猫头鹰,包豪斯_000](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/033_梵高猫头鹰,包豪斯_000.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 一只海鸥和史蒂文·西格正在进行一场凝视比赛，绘画,包豪斯       | 梵高猫头鹰,包豪斯                                            |





### 概念艺术

| ![079_4k专业HDR-DnD幻想概念艺术一条由闪电制成的令人敬畏的龙,故障艺术_004](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/079_4k专业HDR-DnD幻想概念艺术一条由闪电制成的令人敬畏的龙,故障艺术_004.jpg) | ![043_4k专业HDR-DnD奇幻概念艺术小鸡施展幻觉咒语,故障艺术_003](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/043_4k专业HDR-DnD奇幻概念艺术小鸡施展幻觉咒语,故障艺术_003.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 4k专业HDR-DnD幻想概念艺术一条由闪电制成的令人敬畏的龙,概念艺术 | 4k专业HDR-DnD奇幻概念艺术小鸡施展幻觉咒语,概念艺术           |



### 像素艺术

| ![pixel1](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/pixel1.jpg) | ![pixel2](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/pixel2.jpg) | ![pixel3](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/pixel3.jpg) | ![pixel4](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/pixel4.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



### 艺术家

| ![001_萨尔瓦多·达利描绘古代文明的超现实主义梦幻油画,写实风格_006](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/001_萨尔瓦多·达利描绘古代文明的超现实主义梦幻油画,写实风格_006.jpg) | ![033_梵高猫头鹰,蒸汽波艺术_001](https://raw.githubusercontent.com/OleNet/YouPromptMe/gh-pages/you-prompt-me/images/033_梵高猫头鹰,蒸汽波艺术_001.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 萨尔瓦多·达利描绘古代文明的超现实主义梦幻油画,写实风格       | 梵高猫头鹰,蒸汽波艺术                                        |




## 附录

### 常见的艺术家和艺术风格整理

| 艺术类型   | 艺术家                 | 常用艺术风格           |
| ---------- | ---------------------- | ---------------------- |
| 肖像画     | 文森特·梵高            | 印象主义               |
| 风景画     | 尼古拉斯·罗伊里奇      | 现实主义               |
| 风俗画     | 皮埃尔-奥古斯特·雷诺阿 | 浪漫主义               |
| 宗教绘画   | 克劳德·莫内            | 表现主义               |
| 抽象画     | 彼得·孔查洛夫斯基      | 后印象主义             |
| 都市风景画 | 卡米尔·毕沙罗          | 象征主义               |
| 素描与草图 | 约翰·辛格·萨金特       | 新艺术主义             |
| 静物       | 伦勃朗                 | 巴洛克风格             |
| 裸体画     | 马克·夏加尔            | 抽象表现主义           |
| 插画       | 巴勃罗·毕加索          | 北欧文艺复兴           |
|            | 古斯塔夫·多雷          | 素人艺术，原始主义     |
|            | 阿尔布雷特·丢勒        | 立体主义               |
|            | 鲍里斯·库斯妥基耶夫    | 洛可可                 |
|            | 埃德加·德加            | 色域绘画               |
|            |                        | 波普艺术               |
|            |                        | 文艺复兴开端           |
|            |                        | 文艺复兴全盛期         |
|            |                        | 极简主义               |
|            |                        | 矫饰主义，文艺复兴晚期 |



### 常见的摄影风格词整理

| 可以加入到 Prompt 中的摄影词 |              |
| ---------------------------- | ------------ |
| 浅景深                       | 仰拍         |
| 负像                         | 动态模糊     |
| 微距                         | 高反差       |
| 双色版                       | 中心构图     |
| 角度                         | 逆光         |
| 三分法                       | 长曝光       |
| 抓拍                         | 禅宗摄影     |
| 软焦点                       | 抽象微距镜头 |
| 黑白                         | 暗色调       |
| 无镜反射                     | 长时间曝光   |
| 双色调                       | 框架，取景   |
| 颗粒图像                     |              |



<span id = "related-work">   </span>
### 相关链接

美学相关的词汇： https://aesthetics.fandom.com/wiki/List_of_Aesthetics

DALL-E 2 的 Prompt 技巧资料：https://docs.google.com/document/d/11WlzjBT0xRpQhP9tFMtxzd0q6ANIdHPUBkMV-YB043U/edit

DiscoDiffusion Prompt 技巧资料：https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit
