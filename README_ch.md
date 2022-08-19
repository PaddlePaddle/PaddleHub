简体中文 | [English](README.md)

<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle">
<p align="center">
<div align="center">  
  <h3> <a href=#QuickStart> 快速开始 </a> | <a href="https://paddlehub.readthedocs.io/zh_CN/release-v2.1//"> 教程文档 </a> | <a href="./modules/README_ch.md"> 模型库 </a> | <a href="https://www.paddlepaddle.org.cn/hub"> 演示Demo </a>
  </h3>
</div>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleHub/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleHub?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/pypi/format/paddlehub?color=c77"></a>
</p>
<p align="center">
    <a href="https://github.com/PaddlePaddle/PaddleHub/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PaddleHub?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PaddleHub/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/PaddleHub?color=3af"></a>
    <a href="https://pypi.org/project/paddlehub/"><img src="https://img.shields.io/pypi/dm/paddlehub?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleHub/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PaddleHub?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PaddleHub/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleHub?color=ccf"></a>
</p>




## 简介与特性
- PaddleHub旨在为开发者提供丰富的、高质量的、直接可用的预训练模型
- **【模型种类丰富】**: 涵盖大模型、CV、NLP、Audio、Video、工业应用主流六大品类的 **360+** 预训练模型，全部开源下载，离线可运行
- **【超低使用门槛】**：无需深度学习背景、无需数据与训练过程，可快速使用AI模型
- **【一键模型快速预测】**：通过一行命令行或者极简的Python API实现模型调用，可快速体验模型效果
- **【一键模型转服务化】**：一行命令，搭建深度学习模型API服务化部署能力
- **【十行代码迁移学习】**：十行代码完成图片分类、文本分类的迁移学习任务
- **【跨平台兼容性】**：可运行于Linux、Windows、MacOS等多种操作系统

## 近期更新
- **🔥2022.08.19:** 发布v2.3.0版本新增[文心大模型](https://wenxin.baidu.com/)和disco diffusion(dd)系列文图生成模型。
   - 支持对[文心大模型API](https://wenxin.baidu.com/moduleApi)的调用, 包括 文图生成模型**ERNIE-ViLG**([体验Demo](https://aistudio.baidu.com/aistudio/projectdetail/4445016)), 以及支持写作文、写文案、写摘要、对对联、自由问答、写小说、补全文本等多个应用的语言模型**ERNIE 3.0 Zeus**([体验Demo](https://aistudio.baidu.com/aistudio/projectdetail/4445054))。
   - 新增基于disco diffusion技术的文图生成dd系列模型5个，其中英文模型([体验Demo](https://aistudio.baidu.com/aistudio/projectdetail/4444984))3个，中文模型2个。欢迎点击链接在aistudio上进行体验基于**ERNIE-ViL**开发的中文文图生成模型disco_diffusion_ernievil_base([体验Demo](https://aistudio.baidu.com/aistudio/projectdetail/4444998))。
- **2022.02.18:** 加入Huggingface，创建了PaddlePaddle的空间并上传了模型: [PaddlePaddle Huggingface](https://huggingface.co/PaddlePaddle)。

- **🔥2021.12.22**，发布v2.2.0版本新增[预训练模型库官网](https://www.paddlepaddle.org.cn/hublist)。
   - 新增100+高质量模型，涵盖对话、语音处理、语义分割、文字识别、文本处理、图像生成等多个领域，预训练模型总量达到【360+】；
   - 新增模型[检索列表](./modules/README_ch.md)，包含模型名称、网络、数据集和使用场景等信息，快速定位用户所需的模型；
   - 模型文档排版优化，呈现数据集、指标、模型大小等更多实用信息。


- [More](./docs/docs_ch/release.md)



## **精品模型效果展示[【更多】](./docs/docs_ch/visualization.md)[【模型库】](./modules/README_ch.md)**

### **[文心大模型](https://www.paddlepaddle.org.cn/hubdetail?name=ernie_vilg&en_category=TextToImage)**
- 包含大模型ERNIE-ViL、ERNIE 3.0 Zeus, 支持文图生成、写作文、写文案、写摘要、对对联、自由问答、写小说、补全文本等多个应用。
<div align="center">
<img src="https://user-images.githubusercontent.com/22424850/185588578-e2d1216b-e797-458d-bc6b-0ccb8e1bd1b9.png"  width = "80%"  />
</div>


### **[图像类（212个）](./modules/README_ch.md#图像)**
- 包括图像分类、人脸检测、口罩检测、车辆检测、人脸/人体/手部关键点检测、人像分割、80+语言文本识别、图像超分/上色/动漫化等
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_all.gif"  width = "530" height = "400" />
</div>

- 感谢CopyRight@[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)、[AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2)、[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[Zhengxia Zou](https://github.com/jiupinjia/SkyAR)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 提供相关预训练模型，训练能力开放，欢迎体验。


### **[文本类（130个）](./modules/README_ch.md#文本)**
- 包括中文分词、词性标注与命名实体识别、句法分析、AI写诗/对联/情话/藏头诗、中文的评论情感分析、中文色情文本审核等
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_all.gif"  width = "640" height = "240" />
</div>

- 感谢CopyRight@[ERNIE](https://github.com/PaddlePaddle/ERNIE)、[LAC](https://github.com/baidu/LAC)、[DDParser](https://github.com/baidu/DDParser)提供相关预训练模型，训练能力开放，欢迎体验。


### **[语音类（15个）](./modules/README_ch.md#语音)**
- ASR语音识别算法，多种算法可选
- 语音识别效果如下:
<div align="center">
<table>
    <thead>
        <tr>
            <th width=250> Input Audio  </th>
            <th width=550> Recognition Result  </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align = "center">
            <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/en.wav" rel="nofollow">
                    <img align="center" src="./docs/imgs/Readme_Related/audio_icon.png" width=250 ></a><br>
            </td>
            <td >I knocked at the door on the ancient side of the building.</td>
            </tr>
            <tr>
            <td align = "center">
            <a href="https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav" rel="nofollow">
                    <img align="center" src="./docs/imgs/Readme_Related/audio_icon.png" width=250></a><br>
            </td>
            <td>我认为跑步最重要的就是给我带来了身体健康。</td>
        </tr>
    </tbody>
</table>
</div>

- TTS语音合成算法，多种算法可选
- 输入：`Life was like a box of chocolates, you never know what you're gonna get.`
- 合成效果如下:
<div align="center">
<table>
    <thead>
    </thead>
    <tbody>
        <tr>
            <th>deepvoice3 </th>
            <th>fastspeech </th>
            <th>transformer</th>
        </tr>
        <tr>
            <th>
            <a href="https://paddlehub.bj.bcebos.com/resources/deepvoice3_ljspeech-0.wav">
            <img src="./docs/imgs/Readme_Related/audio_icon.png" width=250 /></a><br>
            </th>
            <th>
            <a href="https://paddlehub.bj.bcebos.com/resources/fastspeech_ljspeech-0.wav">
            <img src="./docs/imgs/Readme_Related/audio_icon.png" width=250 /></a><br>
            </th>
            <th>
            <a href="https://paddlehub.bj.bcebos.com/resources/transformer_tts_ljspeech-0.wav">
            <img src="./docs/imgs/Readme_Related/audio_icon.png" width=250 /></a><br>
            </th>
        </tr>
    </tbody>
</table>
</div>

- 感谢CopyRight@[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)提供预训练模型，训练能力开放，欢迎体验。

### **[视频类（8个）](./modules/README_ch.md#视频)**
- 包含短视频分类，支持3000+标签种类，可输出TOP-K标签，多种算法可选。
- 感谢CopyRight@[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo)提供预训练模型，训练能力开放，欢迎体验。
- `举例：输入一段游泳的短视频，算法可以输出"游泳"结果`
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Video.gif"  width = "400" height = "400" />
</div>




##  ===划重点===
- 以上所有预训练模型全部开源，模型数量持续更新，欢迎**⭐Star⭐**关注。
<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleHub/stargazers">
            <img src="./docs/imgs/Readme_Related/star.png"  width = "411" height = "100" /></a>  
</div>

<a name="欢迎加入PaddleHub技术交流群"></a>
## 欢迎加入PaddleHub技术交流群
- 在使用模型过程中有任何问题，可以加入官方微信群，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。
<div align="center">
<img src="./docs/imgs/joinus.PNG"  width = "200" height = "200" />
</div>  
扫码备注"Hub"加好友之后，再发送“Hub”，会自动邀请您入群。  

<div id="QuickStart">




## 快速开始

[【零基础windows安装并实现图像风格迁移】](./docs/docs_ch/get_start/windows_quickstart.md)

[【零基础mac安装并实现图像风格迁移】](./docs/docs_ch/get_start/mac_quickstart.md)

[【零基础linux安装并实现图像风格迁移】](./docs/docs_ch/get_start/linux_quickstart.md)

### 快速安装相关组件
</div>

```python
!pip install --upgrade paddlepaddle -i https://mirror.baidu.com/pypi/simple
!pip install --upgrade paddlehub -i https://mirror.baidu.com/pypi/simple
```

### 极简中文分词案例  
</div>

```python
import paddlehub as hub

lac = hub.Module(name="lac")
test_text = ["今天是个好天气。"]

results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
print(results)
#{'word': ['今天', '是', '个', '好天气', '。'], 'tag': ['TIME', 'v', 'q', 'n', 'w']}
```

### 一行代码部署lac（词法分析）模型
</div>

```python
!hub serving start -m lac
```

 欢迎用户通过[模型搜索](https://www.paddlepaddle.org.cn/hublist)发现更多实用的预训练模型！

 更多迁移学习能力可以参考[教程文档](https://paddlehub.readthedocs.io/zh_CN/release-v2.1/transfer_learning_index.html)




<a name="许可证书"></a>
## 许可证书
本项目的发布受<a href="./LICENSE">Apache 2.0 license</a>许可认证。

<a name="致谢"></a>
## 致谢开发者

<p align="center">
    <a href="https://github.com/nepeplwu"><img src="https://avatars.githubusercontent.com/u/45024560?v=4" width=75 height=75></a>
    <a href="https://github.com/Steffy-zxf"><img src="https://avatars.githubusercontent.com/u/48793257?v=4" width=75 height=75></a>
    <a href="https://github.com/ZeyuChen"><img src="https://avatars.githubusercontent.com/u/1371212?v=4" width=75 height=75></a>
    <a href="https://github.com/ShenYuhan"><img src="https://avatars.githubusercontent.com/u/28444161?v=4" width=75 height=75></a>
    <a href="https://github.com/kinghuin"><img src="https://avatars.githubusercontent.com/u/11913168?v=4" width=75 height=75></a>
    <a href="https://github.com/grasswolfs"><img src="https://avatars.githubusercontent.com/u/23690325?v=4" width=75 height=75></a>
    <a href="https://github.com/haoyuying"><img src="https://avatars.githubusercontent.com/u/35907364?v=4" width=75 height=75></a>
    <a href="https://github.com/sjtubinlong"><img src="https://avatars.githubusercontent.com/u/2063170?v=4" width=75 height=75></a>
    <a href="https://github.com/KPatr1ck"><img src="https://avatars.githubusercontent.com/u/22954146?v=4" width=75 height=75></a>
    <a href="https://github.com/jm12138"><img src="https://avatars.githubusercontent.com/u/15712990?v=4" width=75 height=75></a>
    <a href="https://github.com/DesmonDay"><img src="https://avatars.githubusercontent.com/u/20554008?v=4" width=75 height=75></a>
    <a href="https://github.com/chunzhang-hub"><img src="https://avatars.githubusercontent.com/u/63036966?v=4" width=75 height=75></a>
    <a href="https://github.com/rainyfly"><img src="https://avatars.githubusercontent.com/u/22424850?v=4" width=75 height=75></a>
    <a href="https://github.com/adaxiadaxi"><img src="https://avatars.githubusercontent.com/u/58928121?v=4" width=75 height=75></a>
    <a href="https://github.com/linjieccc"><img src="https://avatars.githubusercontent.com/u/40840292?v=4" width=75 height=75></a>
    <a href="https://github.com/linshuliang"><img src="https://avatars.githubusercontent.com/u/15993091?v=4" width=75 height=75></a>
    <a href="https://github.com/eepgxxy"><img src="https://avatars.githubusercontent.com/u/15946195?v=4" width=75 height=75></a>
    <a href="https://github.com/paopjian"><img src="https://avatars.githubusercontent.com/u/20377352?v=4" width=75 height=75></a>
    <a href="https://github.com/zbp-xxxp"><img src="https://avatars.githubusercontent.com/u/58476312?v=4" width=75 height=75></a>
    <a href="https://github.com/houj04"><img src="https://avatars.githubusercontent.com/u/35131887?v=4" width=75 height=75></a>
    <a href="https://github.com/Wgm-Inspur"><img src="https://avatars.githubusercontent.com/u/89008682?v=4" width=75 height=75></a>
    <a href="https://github.com/AK391"><img src="https://avatars.githubusercontent.com/u/81195143?v=4" width=75 height=75></a>
    <a href="https://github.com/apps/dependabot"><img src="https://avatars.githubusercontent.com/in/29110?v=4" width=75 height=75></a>
    <a href="https://github.com/dxxxp"><img src="https://avatars.githubusercontent.com/u/15886898?v=4" width=75 height=75></a>
    <a href="https://github.com/jianganbai"><img src="https://avatars.githubusercontent.com/u/50263321?v=4" width=75 height=75></a>
    <a href="https://github.com/1084667371"><img src="https://avatars.githubusercontent.com/u/50902619?v=4" width=75 height=75></a>
    <a href="https://github.com/Channingss"><img src="https://avatars.githubusercontent.com/u/12471701?v=4" width=75 height=75></a>
    <a href="https://github.com/Austendeng"><img src="https://avatars.githubusercontent.com/u/16330293?v=4" width=75 height=75></a>
    <a href="https://github.com/BurrowsWang"><img src="https://avatars.githubusercontent.com/u/478717?v=4" width=75 height=75></a>
    <a href="https://github.com/cqvu"><img src="https://avatars.githubusercontent.com/u/37096589?v=4" width=75 height=75></a>
    <a href="https://github.com/DeepGeGe"><img src="https://avatars.githubusercontent.com/u/51083814?v=4" width=75 height=75></a>
    <a href="https://github.com/Haijunlv"><img src="https://avatars.githubusercontent.com/u/28926237?v=4" width=75 height=75></a>
    <a href="https://github.com/holyseven"><img src="https://avatars.githubusercontent.com/u/13829174?v=4" width=75 height=75></a>
    <a href="https://github.com/MRXLT"><img src="https://avatars.githubusercontent.com/u/16594411?v=4" width=75 height=75></a>
    <a href="https://github.com/cclauss"><img src="https://avatars.githubusercontent.com/u/3709715?v=4" width=75 height=75></a>
    <a href="https://github.com/hu-qi"><img src="https://avatars.githubusercontent.com/u/17986122?v=4" width=75 height=75></a>
    <a href="https://github.com/itegel"><img src="https://avatars.githubusercontent.com/u/8164474?v=4" width=75 height=75></a>
    <a href="https://github.com/jayhenry"><img src="https://avatars.githubusercontent.com/u/4285375?v=4" width=75 height=75></a>
    <a href="https://github.com/hlmu"><img src="https://avatars.githubusercontent.com/u/30133236?v=4" width=75 height=75></a>
    <a href="https://github.com/shinichiye"><img src="https://avatars.githubusercontent.com/u/76040149?v=4" width=75 height=75></a>
    <a href="https://github.com/will-jl944"><img src="https://avatars.githubusercontent.com/u/68210528?v=4" width=75 height=75></a>
    <a href="https://github.com/yma-admin"><img src="https://avatars.githubusercontent.com/u/40477813?v=4" width=75 height=75></a>
    <a href="https://github.com/zl1271"><img src="https://avatars.githubusercontent.com/u/22902089?v=4" width=75 height=75></a>
    <a href="https://github.com/brooklet"><img src="https://avatars.githubusercontent.com/u/1585799?v=4" width=75 height=75></a>
    <a href="https://github.com/wj-Mcat"><img src="https://avatars.githubusercontent.com/u/10242208?v=4" width=75 height=75></a>
</p>

我们非常欢迎您为PaddleHub贡献代码，也十分感谢您的反馈。

* 非常感谢[肖培楷](https://github.com/jm12138)贡献了街景动漫化，人像动漫化、手势关键点识别、天空置换、深度估计、人像分割等module
* 非常感谢[Austendeng](https://github.com/Austendeng)贡献了修复SequenceLabelReader的pr
* 非常感谢[cclauss](https://github.com/cclauss)贡献了优化travis-ci检查的pr
* 非常感谢[奇想天外](http://www.cheerthink.com/)贡献了口罩检测的demo
* 非常感谢[mhlwsk](https://github.com/mhlwsk)贡献了修复序列标注预测demo的pr
* 非常感谢[zbp-xxxp](https://github.com/zbp-xxxp)和[七年期限](https://github.com/1084667371)联合贡献了看图写诗中秋特别版module、谣言预测、请假条生成等module
* 非常感谢[livingbody](https://github.com/livingbody)贡献了基于PaddleHub能力的风格迁移和中秋看图写诗微信小程序
* 非常感谢[BurrowsWang](https://github.com/BurrowsWang)修复Markdown表格显示问题
* 非常感谢[huqi](https://github.com/hu-qi)修复了readme中的错别字
* 非常感谢[parano](https://github.com/parano)、[cqvu](https://github.com/cqvu)、[deehrlic](https://github.com/deehrlic)三位的贡献与支持
* 非常感谢[paopjian](https://github.com/paopjian)修改了中文readme模型搜索指向的的网站地址错误[#1424](https://github.com/PaddlePaddle/PaddleHub/issues/1424)
* 非常感谢[Wgm-Inspur](https://github.com/Wgm-Inspur)修复了readme中的代码示例问题，并优化了文本分类、序列标注demo中的RNN示例图
* 非常感谢[zl1271](https://github.com/zl1271)修复了serving文档中的错别字
* 非常感谢[AK391](https://github.com/AK391)在Hugging Face spaces中添加了UGATIT和deoldify模型的web demo
* 非常感谢[itegel](https://github.com/itegel)修复了快速开始文档中的错别字
* 非常感谢[AK391](https://github.com/AK391)在Hugging Face spaces中添加了Photo2Cartoon模型的web demo
