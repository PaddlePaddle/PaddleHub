English | [简体中文](README_ch.md)

<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle"
</p>


------------------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleHub.svg)](https://github.com/PaddlePaddle/PaddleHub/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

## Introduction
- PaddleHub aims to provide developers with rich, high-quality, and directly usable pre-trained models.
- **No need for deep learning background**，you can use AI models quickly and enjoy the dividends of the artificial intelligence era.
- Covers 4 major categories of CV, NLP, Audio, and Video, and supports **one-click prediction**, **one-click service deployment** and **transfer learning**
- All models are **OPEN SOURCE**, **FREE** for download and use in offline scenario.

**Recent updates**
- 2020.12.1: Release 2.0-beta1 version, migrate ERNIE, RoBERTa, BERT to dynamic graph mode. Add text classification fine-tune task based on large-scale pre-trained models.
- 2020.11.20: Release 2.0-beta version, fully migrate the dynamic graph programming mode, and upgrade the service deployment Serving capability; add 1 hand key point detection model, 12 image animation models, 3 image editing models, 3 speech synthesis models, syntax Analyzing one, the total number of pre-trained models reaches **【182】**.
- 2020.10.09: Added 4 new OCR multi-language series models, 4 image editing models, and the total number of pre-trained models reached **【162】**.
- 2020.09.27: 6 new text generation models and 1 image segmentation model were added, and the total number of pre-trained models reached **【154】**.
- 2020.08.13: Released v1.8.1, added a segmentation model, and supports EMNLP2019-Sentence-BERT as a text matching task network. The total number of pre-training models reaches **【147】**.
- 2020.07.29: Release v1.8.0, new AI couplets and AI writing poems, jieba word cutting, text data LDA, semantic similarity calculation, new target detection, short video classification model, ultra-lightweight Chinese and English OCR, new pedestrian detection, vehicle Industrial-grade models such as detection and animal recognition support [VisualDL](https://github.com/PaddlePaddle/VisualDL) visualization training, and the total number of pre-training models reaches **【135】**.


## Features
- **Abundant Pre-trained Models**: 180+ pre-trained models covering the four major categories of CV, NLP, Audio, and Video, all open source and free for downloads and offline usage.
- **Quick Model Prediction**: Model calls can be realized through a one-line command line or a minimalist Python API to quickly experience the model effect.
- **Model As Service**: A one-line command to build deep learning model API service deployment capabilities.
- **Super Easy Transfer Learning**: Just few lines of code you complete the transfer-learning task like image classification and text classification, etc.
- **Cross-platform**: Can run on Linux, Windows, MacOS and other operating systems.

## Visualization Demo

### Text Recognition
- Contains ultra-lightweight Chinese and English OCR models, high-precision Chinese and English, multilingual German, French, Japanese, Korean OCR recognition.
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_Ocr.gif"  width = "800" height = "400" />
</div>

### Face Detection
- Including face detection, mask face detection, multiple algorithms are optional.
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_ObjectDetection_Face_Mask.gif"  width = "588" height = "400" />
</div>

### Image Editing
- 4x super resolution effect, multiple super resolution models are optional.
- Colorization models can be used to repair old grayscale photos.

<div align="center">
<table>
    <thead>
    </thead>
    <tbody>
        <tr>
            <th>SuperResolution </th>
            <th>Restoration </th>
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

### Object Detection
- Pedestrian detection, vehicle detection, and more industrial-grade ultra-large-scale pretrained models are provided.
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_ObjectDetection_Pedestrian_Vehicle.gif"  width = "642" height = "400" />
</div>

### Key Point Detection
- Supports body, face and hands key point detection for single or multiple person.
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_keypoint.gif"  width = "458" height = "400" />
</div>

### Image Segmentation
- High quality pixel-level portrait cutout model, ACE2P human body analysis world champion models are provided.
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageSeg_Human.gif"  width = "642" height = "400" />
</div>

### Image Animation
- Image style transfer models with Hayao Miyazaki and Makoto Shinkai styles, etc are provided
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageGan_Anime.gif"  width = "642" height = "400" />
</div>

### Image Classification
- Various models like animal classification, dish classification, wild animal product classification are available.
<div align="center">
<img src="./docs/imgs/Readme_Related/ImageClas_animal_dish_wild.gif"  width = "530" height = "400" />
</div>

### Text Generation
- AI poem writing, AI couplets, AI love words generation models are available.
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Textgen_poetry.gif"  width = "850" height = "400" />
</div>

### Syntax Analysis
- Leading Chinese syntactic analysis model release by Baidu NLP.
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_SyntacticAnalysis.png"  width = "640" height = "301" />
</div>

### Sentiment Analysis
- All SOTA Chinese sentiment analysis model released by Baidu NLP can be used just one-line of code.
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_SentimentAnalysis.png"  width = "640" height = "228" />
</div>

### Text Review
- Contains the review of Chinese pornographic text, and multiple algorithms are available.
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Textreview.png"  width = "640" height = "140" />
</div>

### Speech Synthesis
- TTS speech synthesis algorithm, multiple algorithms are available
- Input: `Life was like a box of chocolates, you never know what you're gonna get.`
- The synthesis effect is as follows:
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

### Video Classification
- Short video classification trained via large-scale video dataset, supports 3000+ tag types prediction.
- `Example: Input a short video of swimming, the algorithm can output the result of "swimming"`
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Video.gif"  width = "400" height = "400" />
</div>

## ===Key Points===
- All the above pre-trained models are all open source, and the number of models is continuously updated. Welcome Star to pay attention.
<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleHub/stargazers">
    <img src="./docs/imgs/Readme_Related/star_en.png"  width = "411" height = "100" /></a>  
</div>

<a name="Welcome_joinus"></a>

## Welcome to join PaddleHub technical group

If you have any questions during the use of the model, you can join the official WeChat group to get more efficient questions and answers, and fully communicate with developers from all walks of life. We look forward to your joining.
<div align="center">
<img src="./docs/imgs/joinus.PNG"  width = "200" height = "200" />
</div>  
If you fail to scan the code, please add WeChat 15711058002 and note "Hub", the operating class will invite you to join the group.

## Documentation Tutorial
- [PIP Installation](./docs/docs_en/installation_en.md)
- Quick Start
    - [Command Line](./docs/docs_en/quick_experience/cmd_quick_run_en.md)
    - [Python API](./docs/docs_en/quick_experience/python_use_hub_en.md)
    - [More Demos](./docs/docs_en/quick_experience/more_demos_en.md)
- Rich Pre-trained Models 182
    - [Boutique Featured Models](./docs/docs_en/figures_en.md)
    - Computer Vision 126
      - [Image Classification 64 ](./modules/image/classification/README_en.md)
      - [Object Detection 13 ](./modules/image/object_detection/README_en.md)
      - [Face Detection 7 ](./modules/image/face_detection/README_en.md)  
      - [Key Point Detection 3 ](./modules/image/keypoint_detection/README_en.md)
      - [Image Segmentation 7 ](./modules/image/semantic_segmentation/README_en.md)
      - [Text Recognition 8 ](./modules/image/text_recognition/README_en.md)
      - [Image Generation 17 ](./modules/image/Image_gan/README_en.md)
      - [Image Editing 7 ](./modules/image/Image_editing/README_en.md)
    - Natural Language Processing 48
      - [Lexical Analysis 2 ](./modules/text/lexical_analysis/README_en.md)
      - [Syntax Analysis 1 ](./modules/text/syntactic_analysis/README_en.md)
      - [Sentiment Analysis 7 ](./modules/text/sentiment_analysis/README_en.md)
      - [Text Review 3 ](./modules/text/text_review/README_en.md)
      - [Text Generation 9 ](./modules/text/text_generation/README_en.md)
      - [Semantic Models 26 ](./modules/text/language_model/README_en.md)
    - Audio 3
      - [Speech Synthesis 3 ](./modules/audio/README_en.md)
    - Video 5
      - [Video Classification 5 ](./modules/video/README_en.md)
- Deploy
    - [Local Inference Deployment](./docs/docs_en/quick_experience/python_use_hub_en.md)
    - [One Line of Code Service deployment](./docs/docs_en/tutorial/serving_en.md)
    - [Mobile Device Deployment](https://paddle-lite.readthedocs.io/zh/latest/quick_start/tutorial.html)
- Advanced documentation
    - [Command Line Interface Usage](./docs/docs_en/tutorial/cmdintro_en.md)
    - [How to Load Customized Dataset](./docs/docs_en/tutorial/how_to_load_data_en.md)
- Community
    - [Join Technical Group](#Welcome_joinus)
    - [Contribute Pre-trained Models](./docs/docs_en/contribution/contri_pretrained_model_en.md)
    - [Contribute Code](./docs/docs_en/contribution/contri_pr_en.md)
- [License](#License)
- [Contribution](#Contribution)

<a name="License"></a>
## License
The release of this project is certified by the <a href="./LICENSE">Apache 2.0 license</a>.

<a name="Contribution"></a>
## Contribution
We welcome you to contribute code to PaddleHub, and thank you for your feedback.

* Many thanks to [Austendeng](https://github.com/Austendeng) for fixing the SequenceLabelReader
* Many thanks to [cclauss](https://github.com/cclauss) optimizing travis-ci check
* Many thanks to [奇想天外](http://www.cheerthink.com/)，Contributed a demo of mask detection
* Many thanks to [mhlwsk](https://github.com/mhlwsk)，Contributed the repair sequence annotation prediction demo
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp)，Contributed modules for viewing pictures and writing poems
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp) and [七年期限](https://github.com/1084667371),Jointly contributed to the Mid-Autumn Festival Special Edition Module
* Many thanks to [livingbody](https://github.com/livingbody)，Contributed models for style transfer based on PaddleHub's capabilities and Mid-Autumn Festival WeChat Mini Program
