简体中文 | [English](README.md)

<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle"
</p>

------------------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)


## 简介
- PaddleHub旨在为开发者提供丰富的、高质量的、直接可用的预训练模型。
- **【无需深度学习背景、无需数据与训练过程】**，可快速使用AI模型，享受人工智能时代红利。
- 涵盖CV、NLP、Audio、Video主流四大品类，支持**一键预测**、**一键服务化部署**和**快速迁移学习**
- 全部模型开源下载，**离线可运行**。


## 近期更新
- **2021.02.18**，发布v2.0.0版本，模型开发调试更简单，finetune接口更加灵活易用。视觉类任务迁移学习能力全面升级，支持[图像分类](./demo/image_classification/README.md)、[图像着色](./demo/colorization/README.md)、[风格迁移](./demo/style_transfer/README.md)等多种任务；BERT、ERNIE、RoBERTa等Transformer类模型升级至动态图，支持[文本分类](./demo/text_classification/README.md)、[序列标注](./demo/sequence_labeling/README.md)的Fine-Tune能力；优化服务化部署Serving能力，支持多卡预测、自动负载均衡，性能大幅度提升；新增自动数据增强能力[Auto Augment](./demo/autoaug/README.md)，能高效地搜索适合数据集的数据增强策略组合。新增[词向量模型](./modules/text/embedding)61个，其中包含中文模型51个，英文模型10个；新增[图像分割](./modules/thirdparty/image/semantic_segmentation)模型4个、[深度模型](./modules/thirdparty/image/depth_estimation)2个、[图像生成](./modules/thirdparty/image/Image_gan/style_transfer)模型7个、[文本生成](./modules/thirdparty/text/text_generation)模型3个。预训练模型总量达到[**【274】**](https://www.paddlepaddle.org.cn/hublist) 个。
- **2020.11.20**，发布2.0-beta版本，全面迁移动态图编程模式，服务化部署Serving能力升级；新增手部关键点检测1个、图像动漫化类12个、图片编辑类3个，语音合成类3个，句法分析1个，预训练模型总量到达 **【182】** 个。
- **2020.10.09**，新增OCR多语言系列模型4个，图像编辑模型4个，预训练模型总量到达 **【162】** 个。
- **2020.09.27**，新增文本生成模型6个，图像分割模型1个，预训练模型总量到达 **【154】** 个。
- **2020.08.13**，发布v1.8.1，新增人像分割模型Humanseg，支持EMNLP2019-Sentence-BERT作为文本匹配任务网络，预训练模型总量到达 **【147】** 个。
- **2020.07.29**，发布v1.8.0，新增AI对联和AI写诗、jieba切词，文本数据LDA、语义相似度计算，新增目标检测，短视频分类模型，超轻量中英文OCR，新增行人检测、车辆检测、动物识别等工业级模型，支持VisualDL可视化训练，预训练模型总量到达 **【135】** 个。
- [More](./docs/docs_ch/release.md)


## [特性](./docs/docs_ch/figures.md)
- **【丰富的预训练模型】**：涵盖CV、NLP、Audio、Video主流四大品类的 180+ 预训练模型，全部开源下载，离线可运行。
- **【一键模型快速预测】**：通过一行命令行或者极简的Python API实现模型调用，可快速体验模型效果。
- **【一键模型转服务化】**：一行命令，搭建深度学习模型API服务化部署能力。
- **【十行代码迁移学习】**：十行代码完成图片分类、文本分类的迁移学习任务
- **【跨平台兼容性】**：可运行于Linux、Windows、MacOS等多种操作系统


## 精品模型效果展示

### 文本识别
- 包含超轻量中英文OCR模型，高精度中英文、多语种德语、法语、日语、韩语OCR识别。
- 感谢CopyRight@[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_Ocr.gif"  width = "800" height = "400" />
</div>

### 人脸检测
- 包含人脸检测，口罩人脸检测，多种算法可选。
- 感谢CopyRight@[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_ObjectDetection_Face_Mask.gif"  width = "588" height = "400" />
</div>

### 图像编辑
- 4倍超分效果，多种超分算法可选。
- 黑白图片上色，可用于老旧照片修复，
- 感谢CopyRight@[PaddleGan](https://github.com/PaddlePaddle/PaddleGan)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<table>
    <thead>
    </thead>
    <tbody>
        <tr>
            <th>图像超分辨率 </th>
            <th>黑白图片上色 </th>
        </tr>
        <tr>
            <th>
            <a>
            <img src="./docs/imgs/Readme_Related/ImageEdit_SuperResolution.gif"  width = "266" height = "400" /></a><br>
            </th>
            <th>
            <a>
            <img src="./docs/imgs/Readme_Related/ImageEdit_Restoration.gif"  width = "300" height = "400" /></a><br>
            </th>
        </tr>
    </tbody>
</table>
</div>


### 图像生成
- 包含人像动漫化、街景动漫化、风格迁移。
- 感谢CopyRight@[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)、CopyRight@[AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2)提供预训练模型。
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageGAN.gif"  width = "640" height = "600" />
</div>

### 目标检测
- 包含行人检测、车辆检测，更有工业级超大规模预训练模型可选。--
- 感谢CopyRight@[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_ObjectDetection_Pedestrian_Vehicle.gif"  width = "642" height = "400" />
</div>

### 关键点检测
- 包含单人、多人身体关键点检测、面部关键点检测、手部关键点检测。
- 感谢CopyRight@[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)开源预训练模型。
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_keypoint.gif"  width = "642" height = "550" />
</div>

### 图像分割
- 包含效果卓越的人像抠图模型、ACE2P人体解析世界冠军模型、动态天空置换算法
- 感谢CopyRight@[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、感谢CopyRight@[Zhengxia Zou](https://github.com/jiupinjia/SkyAR)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageSeg_Human.gif"  width = "642" height = "400" />
</div>

<div align="center">
<img src="./docs/imgs/Readme_Related/9dis.gif"  width = "642" height = "200" />
</div>

<div align="center">

（第二张动图来自于CopyRight@[jiupinjia/SkyAR](https://github.com/jiupinjia/SkyAR#district-9-ship-video-source))
</div>

### 图像分类
- 包含动物分类、菜品分类、野生动物制品分类，多种算法可选
- 感谢CopyRight@[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageClas_animal_dish_wild.gif"  width = "530" height = "400" />
</div>

### 文本生成
- 包含AI写诗、AI对联、AI情话、AI藏头诗，多种算法可选。
- 感谢CopyRight@[ERNIE](https://github.com/PaddlePaddle/ERNIE)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Textgen_poetry.gif"  width = "850" height = "400" />
</div>

### 词法分析
- 效果优秀的中文分词、词性标注与命名实体识别的模型。
- 感谢CopyRight@[LAC](https://github.com/baidu/LAC)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Lexical Analysis.png"  width = "640" height = "233" />
</div>

### 句法分析
- 效果领先的中文句法分析模型。
- 感谢CopyRight@[DDParser](https://github.com/baidu/DDParser)提供预训练模型，训练能力开放，欢迎体验。
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_SyntacticAnalysis.png"  width = "640" height = "301" />
</div>

### 情感分析
- 支持中文的评论情感分析
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_SentimentAnalysis.png"  width = "640" height = "228" />
</div>

### 文本审核
- 包含中文色情文本的审核，多种算法可选。
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Textreview.png"  width = "640" height = "140" />
</div>

### 语音合成
- TTS语音合成算法，多种算法可选
- 感谢CopyRight@[Parakeet](https://github.com/PaddlePaddle/Parakeet)提供预训练模型，训练能力开放，欢迎体验。
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

### 视频分类
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
如扫码失败，请添加微信15704308458，并备注“Hub”，运营同学会邀请您入群。  


## 文档教程

- [PIP安装](./docs/docs_ch/installation.md)
- 快速开始
    - [命令行调用](./docs/docs_ch/quick_experience/cmd_quick_run.md)
    - [Python API调用](./docs/docs_ch/quick_experience/python_use_hub.md)
    - [示例体验项目demo](./docs/docs_ch/quick_experience/more_demos.md)
- 丰富的预训练模型 274
    - [精品特色模型](./docs/docs_ch/figures.md)
    - 计算机视觉 141 个
      - [图像分类 64 个](./modules/image/classification/README.md)
      - [目标检测 13 个](./modules/image/object_detection/README.md)
      - [人脸检测 7 个](./modules/image/face_detection/README.md)  
      - [关键点检测 5 个](./modules/image/keypoint_detection/README.md)
      - [图像分割 13 个](./modules/image/semantic_segmentation/README.md)
      - [文本识别 8 个](./modules/image/text_recognition/README.md)
      - [图像生成 22 个](./modules/image/Image_gan/README.md)
      - [图像编辑 9 个](./modules/image/Image_editing/README.md)
    - 自然语言处理 122 个
      - [词法分析 2 个](./modules/text/lexical_analysis/README.md)
      - [句法分析 1 个](./modules/text/syntactic_analysis/README.md)
      - [情感分析 7 个](./modules/text/sentiment_analysis/README.md)
      - [文本审核 3 个](./modules/text/text_review/README.md)
      - [文本生成 12 个](./modules/text/text_generation/README.md)
      - [语义模型 36 个](./modules/text/language_model/README.md)
      - [词向量 61 个](https://www.paddlepaddle.org.cn/hublist)
    - 语音 3 个
      - [语音合成 3 个](./modules/audio/README.md)
    - 视频8个
      - [视频分类 5 个](./modules/video/README.md)
      - [视频修复 3 个](https://www.paddlepaddle.org.cn/hublist)
- 部署
    - [本地Inference部署](./docs/docs_ch/quick_experience/python_use_hub.md)
    - [一行代码服务化部署](./docs/docs_ch/tutorial/serving.md)
    - [移动端 Lite 部署（跳转Lite教程）](https://paddle-lite.readthedocs.io/zh/latest/quick_start/tutorial.html)
- 进阶文档
    - [命令行工具详解](./docs/docs_ch/tutorial/cmdintro.md)
    - [自定义数据迁移学习](./docs/docs_ch/tutorial/how_to_load_data.md)
- 社区交流
    - [加入技术交流群](#欢迎加入PaddleHub技术交流群)
    - [贡献预训练模型](./docs/docs_ch/contribution/contri_pretrained_model.md)
    - [贡献代码](./docs/docs_ch/contribution/contri_pr.md)
- [FAQ](./docs/docs_ch/faq.md)  
- [更新历史](./docs/docs_ch/release.md)
- [许可证书](#许可证书)
- [致谢](#致谢)


<a name="许可证书"></a>
## 许可证书
本项目的发布受<a href="./LICENSE">Apache 2.0 license</a>许可认证。

<a name="致谢"></a>
## 致谢开发者
我们非常欢迎您为PaddleHub贡献代码，也十分感谢您的反馈。

* 非常感谢[肖培楷](https://github.com/jm12138)贡献了街景动漫化，人像动漫化、手势关键点识别、天空置换、深度估计、人像分割等module
* 非常感谢[Austendeng](https://github.com/Austendeng)贡献了修复SequenceLabelReader的pr
* 非常感谢[cclauss](https://github.com/cclauss)贡献了优化travis-ci检查的pr
* 非常感谢[奇想天外](http://www.cheerthink.com/)贡献了口罩检测的demo
* 非常感谢[mhlwsk](https://github.com/mhlwsk)贡献了修复序列标注预测demo的pr
* 非常感谢[zbp-xxxp](https://github.com/zbp-xxxp)和[七年期限](https://github.com/1084667371)联合贡献了看图写诗中秋特别版module、谣言预测、请假条生成等module
* 非常感谢[livingbody](https://github.com/livingbody)贡献了基于PaddleHub能力的风格迁移和中秋看图写诗微信小程序
* 非常感谢[BurrowsWang](https://github.com/BurrowsWang)修复Markdown表格显示问题
