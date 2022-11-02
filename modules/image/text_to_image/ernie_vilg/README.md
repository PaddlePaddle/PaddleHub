  <p align="center">
    <img src="https://user-images.githubusercontent.com/22424850/187387422-f6c9ccab-7fda-416e-a24d-7d6084c46f67.jpg"  width = "80%" hspace='10'/>

# PaddleHub ERNIE-ViLG

# 目录
1. [模型基本信息](#一模型基本信息)
2. [安装](#二安装)
3. [模型API预测](#三模型api预测)
4. [Prompt 指南](#四-prompt-指南)
5. [服务部署](#五服务部署)
6. [更新历史](#六更新历史)


## 一、模型基本信息

|模型名称|ernie_vilg|
| :--- | :---: |
|类别|图像-文图生成|
|网络|ERNIE-ViLG|
|数据集|-|
|是否支持Fine-tuning|否|
|模型大小|-|
|最新更新日期|2022-10-14|
|数据指标|-|

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
              style: Optional[str] = "探索无限",
              resolution: Optional[str] = "1024*1024",
              topk: Optional[int] = 6,
              output_dir: Optional[str] = 'ernievilg_output')
    ```

    - 文图生成API，生成文本描述内容的图像。

    - **参数**

      - text_prompts(str): 输入的语句，描述想要生成的图像的内容。
      - style(Optional[str]): 生成图像的风格，当前支持 古风、油画、水彩、卡通、二次元、浮世绘、蒸汽波艺术、
        low poly、像素风格、概念艺术、未来主义、赛博朋克、写实风格、洛丽塔风格、巴洛克风格、超现实主义、探索无限。
      - resolution(Optional[str]): 生成图像的分辨率，当前支持 '1024\*1024', '1024\*1536', '1536\*1024'，默认为'1024\*1024'。
      - topk(Optional[int]): 保存前多少张图，最多保存6张。
      - output_dir(Optional[str]): 保存输出图像的目录，默认为"ernievilg_output"。


    - **返回**
      - images(List(PIL.Image)): 返回生成的所有图像列表，PIL的Image格式。


## 四、 Prompt 指南

作者：佳祥 (LCL-Brew) & 单斌

### Prompt公式

「公式」= 图片主体，细节词，修饰词
细节词可以任意组合，修饰词可以限定一种风格，也可以限定多种风格，遵循的基本原则是符合正常的中文语法逻辑即可。

### 示例

|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/174_蒙娜丽莎，赛博朋克，宝丽来，33毫米,蒸汽波艺术_000-1_7b4a78a.png" alt="drawing" width="300"/>|
| --- |
| prompt：蒙娜丽莎，赛博朋克，宝丽来，33毫米,</br>蒸汽波艺术  |


|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/3_72d9343.png" alt="drawing" width="300"/>|
| --- |
| prompt：火焰，凤凰，少女，未来感，高清，3d，</br>精致面容，cg感，古风，唯美，毛发细致，上半身立绘 |


|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/4_e1f5cbb.png" alt="drawing" width="300"/>|
| --- |
|  prompt：巨狼，飘雪，蓝色大片烟雾，毛发细致，</br>烟雾缭绕，高清，3d，cg感，侧面照  |


| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/5_d380451.png" alt="drawing" width="400"/> |
| --- |
|  <center>prompt：浮世绘日本科幻哑光绘画，概念艺术，</br>动漫风格神道寺禅园英雄动作序列，包豪斯</center> |

### 修饰词

好的修饰词可以让图片生成的效果更好，基于产业级知识增强的文心大模型，用户可以通过输入独特且特征明显的修饰词，来达到更高质量的图片生成。

#### 1. 效果参考
<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/1_3612449.jpg" alt="drawing" width="600"/>

**cg感**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/2_b72fd7a.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/image%20%281%29_8a6b56b.png" alt="drawing" width="300"/> |
| --- | --- |


|  <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/6_2363c54.png" alt="drawing" width="300"/>|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/7_a0910bf.png" alt="drawing" width="300"/>|
| --- | --- |


**厚涂风格 / 厚涂版绘**



| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/8_ea9d4f2.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/9_8defb0a.png" alt="drawing" width="300"/> |
| --- | --- |

|  <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/10_328c202.png" alt="drawing" width="300"/>|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/11_748702c.png" alt="drawing" width="300"/>|
| --- | --- |


**古风**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/12_85ba92e.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/13_cec7db5.png" alt="drawing" width="300"/> |
| --- | --- |


|  <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/14_3511a5d.png" alt="drawing" width="300"/>|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/15_2443b20.png" alt="drawing" width="300"/>|
| --- | --- |


**精致面容**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/16_c79ef20.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/17_9334d56.png" alt="drawing" width="300"/> |
| --- | --- |

|  <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/18_4ba96b0.png" alt="drawing" width="300"/>|<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/19_e627e62.png" alt="drawing" width="300"/>|
| --- | --- |


**穆夏 / 穆夏风格**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/20_2cd8cfb.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/21_75b47a2.png" alt="drawing" width="300"/> |
| --- | --- |

**机械感 / 机械**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/22_c43e94f.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/23_3e85390.png" alt="drawing" width="300"/> |
| --- | --- |

**宫崎骏动画**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/24_02e9187.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/25_ecca869.png" alt="drawing" width="300"/> |
| --- | --- |


**烟雾 / 烟雾缭绕**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/26_d2bf84c.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/27_c482d21.png" alt="drawing" width="300"/> |
| --- | --- |

**皮克斯动画**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/28_b15c2c3.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/29_1a87854.png" alt="drawing" width="300"/> |
| --- | --- |

**拟人化**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/30_f55ea2d.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/31_bc12eaa.png" alt="drawing" width="300"/> |
| --- | --- |

**剪纸叠加风格**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/32_60f30a6.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/33_c4020cc.png" alt="drawing" width="300"/> |
| --- | --- |

**色彩斑斓**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/34_16c64b5.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/35_4b439ff.png" alt="drawing" width="300"/> |
| --- | --- |

**城市印象 & 圆形轮廓**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/36_2ed177e.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/37_d00e2bc.png" alt="drawing" width="300"/> |
| --- | --- |

**上半身立绘 / 人物立绘**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/38_0ec9be4.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/39_e72f64a.png" alt="drawing" width="300"/> |
| --- | --- |

**电影质感**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/40_f90ee02.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/41_1d3da07.png" alt="drawing" width="300"/> |
| --- | --- |

**扁平化设计 / 扁平化**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/42_a6fe543.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/43_360f7d8.png" alt="drawing" width="300"/> |
| --- | --- |

**logo设计 / 简约logo设计**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/44_73b7e12.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/45_5d2b093.png" alt="drawing" width="300"/> |
| --- | --- |

**细节清晰**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/46_e9e50e1.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/47_396cba1.png" alt="drawing" width="300"/> |
| --- | --- |

**毛发细致**

| <img src="https://bce.bdstatic.com/doc/AIDP/wenxin/48_55f90be.png" alt="drawing" width="300"/> |<img src="https://bce.bdstatic.com/doc/AIDP/wenxin/49_4b86ed3.png" alt="drawing" width="300"/> |
| --- | --- |





#### 2. 风格词参考


复古未来主义风格 ->- 海滩兔风格 ->- 抽象技术风格 ->- 酸性精灵风格 ->- 古埃及风格 ->- 风帽风格 ->- 装饰艺术风格 ->- 极光风格 ->- 秋天风格 ->- 巴洛克风格 ->- 摩托车手风格 ->- 碎核风格 ->- 纸箱风格 ->- 未来主义风格 ->- 孟菲斯公司风格 ->- 立体主义风格 ->-赛博朋克风格 ->- 黑暗自然主义风格 ->- 表现主义风格 ->- 野兽派风格 ->- 鬼魂风格 ->- 嘻哈风格 ->- 嬉皮士风格 ->- 幻象之城风格 ->- 印象主义风格 ->- 卡瓦伊风格 ->- 美人鱼风格 ->- 极简主义风格 ->- 水井惠郎风格 ->- 苔藓风格 ->- 新浪潮风格 ->- 迷宫物语风格 ->- 仙女风格 ->- 粉彩朋克风格 ->- 照片写实风格 ->- 粉红公主风格 ->- 海盗风格 ->- 像素可爱风格 ->- 波普艺术风格 ->- 史前遗迹风格 ->- 迷幻风格 ->- 雨天风格 ->- 湿漉漉的风格 ->- 浮世绘风格 ->- 矢量心风格 ->- 维京人风格 ->- 女巫店风格 ->- 后印象主义 ->- 素人主义

#### 3. 艺术词参考



| &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;艺术类型&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;艺术家&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;常用艺术风格&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --- | --- | --- |
| <center>肖像画| 	<center>文森特·梵高| 	<center>印象主义|
| <center>风景画	| <center>尼古拉斯·罗伊里奇	| <center>现实主义|
| <center>风俗画	| <center>皮埃尔-奥古斯特·雷诺阿| 	<center>浪漫主义|
| <center>宗教绘画	| <center>克劳德·莫内	| <center>表现主义|
| <center>抽象画| 	<center>彼得·孔查洛夫斯基	| <center>后印象主义|
| <center>都市风景画| 	<center>卡米尔·毕沙罗	| <center>象征主义|
| <center>素描与草图| 	<center>约翰·辛格·萨金特| 	<center>新艺术主义|
| <center>静物| 	<center>伦勃朗| 	<center>巴洛克风格|
| <center>裸体画| 	<center>马克·夏加尔| 	<center>抽象表现主义|
| <center>插画| 	<center>巴勃罗·毕加索	| <center>北欧文艺复兴|
| | <center>古斯塔夫·多雷	| <center>素人艺术，原始主义|
| | <center>阿尔布雷特·丢勒	| <center>立体主义|
| | <center>鲍里斯·库斯妥基耶夫	| <center>洛可可|
| | <center>埃德加·德加| 	<center>色域绘画|
| | |  <center>波普艺术|
| | | <center>文艺复兴开端|  
| | | <center>文艺复兴全盛期| |
|| |  <center>极简主义|
| | | <center>矫饰主义，文艺复兴晚期|



#### 4. 摄影词参考


|<center>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;可以加入到Prompt 中的摄影词&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|
| --- | --- |
|<center>浅景深	|<center>仰拍|
|<center>负像	|<center>动态模糊|
|<center>微距	|<center>高反差|
|<center>双色版|	<center>中心构图|
|<center>角度	|<center>逆光|
|<center>三分法|	<center>长曝光|
|<center>抓拍	|<center>禅宗摄影|
|<center>软焦点|	<center>抽象微距镜头|
|<center>黑白|	<center>暗色调|
|<center>无镜反射|	<center>长时间曝光|
|<center>双色调|	<center>框架，取景|
|<center>颗粒图像||



### 技巧提示

1. 【作图规则】Prompt构建是文本符合逻辑的组合，有序且丰富的描述可以不断提升画面效果
2. 【新手入门】不知如何输入Prompt？点击示例，体验文生图的魅力，参考教程，逐步进阶~
3. 【风格生成】试试添加 “国潮”、“国风”等，感受中国风的魅力
4. 【风格生成】试试混合两种代表性的风格，例如“赛博朋克，扁平化设计”、”皮克斯动画，赛博朋克”
5. 【人像生成】添加“仙鹤、月亮、楼阁、小屋、街道、玫瑰、机械”，画面会更饱满
6. 【人像生成】添加“精致面容、唯美、cg感、细节清晰“等，人物刻画会更细致
7. 【风格生成】添加“扁平化风格，logo”等，可以设计出各类图标等，例如 “猫猫头像，扁平化风格”
8. 【风格生成】指定颜色，或添加“烟雾缭绕”、“火焰”、“烟尘”、“花瓣”，生成画面的氛围感更饱满
9. 【创意生成】发挥想象力，例如：“中西混搭”、“泰迪熊唱京剧”、“米老鼠吃火锅”
10. 【风格生成】“水彩”，“水墨”与古诗组合，画面意境会有提升~
11. 【风格生成】想要日系头像和拟人化动物？试试关键词“日系手绘”、“治愈风”
12. 【风格生成】添加“pixiv”，生成二次元或者动漫的画质更惊艳

### 呼吁与准则

利用AI技术生成图片的最终目的是要便捷地为人类创造美的作品，激发人的想象力和创作力。而技术在发展中，做不到十全十美，不能保证每次生成的图片都能够尽善尽美。因此呼吁所有用户，您想分享满意的AI图片时，请以正能量进行传播宣传！
算法生成的图片难免会受到数据的影响，从而导致生成的图片是有数据偏见的。因此在分享AI生成图片到社交媒体之前，请谨慎评估当前的图片是不是含有：令人不适的、暴力的、色情的内容。如对以上的内容进行恶意传播，您将会承担相应的法律后果。



<span id = "related-work">   </span>
### 相关链接

美学相关的词汇： https://aesthetics.fandom.com/wiki/List_of_Aesthetics

DALL-E 2 的 Prompt 技巧资料：https://docs.google.com/document/d/11WlzjBT0xRpQhP9tFMtxzd0q6ANIdHPUBkMV-YB043U/edit

DiscoDiffusion Prompt 技巧资料：https://docs.google.com/document/d/1l8s7uS2dGqjztYSjPpzlmXLjl5PM3IGkRWI3IiCuK7g/edit

## 五、服务部署

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


## 六、更新历史

* 1.0.0

  初始发布

* 1.1.0

  增加分辨率参数以及所支持的风格

* 1.2.0

  移除分辨率参数

  ```shell
  $ hub install ernie_vilg == 1.2.0
  ```
