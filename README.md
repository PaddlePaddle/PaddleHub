English | [简体中文](README_ch.md)

<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle" width="400" />
<p align="center">
<div align="center">  
  <h3> <a href=#QuickStart> Quick Start </a> | <a href="./modules"> Model List </a> | <a href=#demos> Demos </a> </h3>
</div>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/pypi/format/paddlehub?color=c77"></a>
    <a href="https://pypi.org/project/paddlehub/"><img src="https://img.shields.io/pypi/dm/paddlehub?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleHub/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleHub?color=ccf"></a>
    <a href="https://huggingface.co/PaddlePaddle"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue"></a>
</p>


## ⭐Features
- **📦400+ AI Models**: Rich, high-quality AI models, including CV, NLP, Speech, Video and Cross-Modal. 
- **🧒Easy to Use**: 3 lines of code to predict 400+ AI models.
- **💁Model As Service**: Easy to serve model with only one line of command.
- **💻Cross-platform**: Support Linux, Windows and MacOS.

### 💥Recent Updates
- **🔥2022.08.19:** The v2.3.0 version is released 🎉
  -  Supports [**ERNIE-ViLG**](./modules/image/text_to_image/ernie_vilg)([HuggingFace Space Demo](https://huggingface.co/spaces/PaddlePaddle/ERNIE-ViLG))
  -  Supports [**Disco Diffusion (DD)**](./modules/image/text_to_image/disco_diffusion_clip_vitb32) and [**Stable Diffusion (SD)**](./modules/image/text_to_image/stable_diffusion)

- **2022.02.18:** Release models to HuggingFace [PaddlePaddle Space](https://huggingface.co/PaddlePaddle)

- For more previous release please refer to [**PaddleHub Release Note**](./docs/docs_en/release.md)


<a name="demos"></a>
## 🌈Visualization Demo





#### 🏜️ [Text-to-Image Models](https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image)
<div align="center">
<table>
    <tr>
        <td><img src="https://user-images.githubusercontent.com/59186797/200235049-fefa7642-6c4c-4f93-bd84-3b36a8a80595.gif"  width = "100%"></td>
        <td><img src="https://user-images.githubusercontent.com/59186797/200244625-77310db8-c9b2-4293-8fe9-c9aae27ee462.gif" width = "80%"></td>
        <td><img src="https://user-images.githubusercontent.com/59186797/200245387-daaf576d-8224-4937-82b8-27e31ee2df16.gif" width = "100%"></td>
    <tr>
    <tr>
        <td align="center"><a href="https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image/ernie_vilg">Wenxin Big Moels</a></td>
        <td align="center"><a href="https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image/stable_diffusion">Stable_Diffusion series</a></td>
        <td align="center"><a href="https://github.com/PaddlePaddle/PaddleHub/tree/develop/modules/image/text_to_image/disco_diffusion_ernievil_base">Disco Diffusion series</a></td>
        
<tr>

<tr>
        <td align="center">Include ERNIE-ViLG, ERNIE-ViL, ERNIE 3.0 Zeus, supports applications such as text-to-image, writing essays, summarization, couplets, question answering, writing novels and completing text。</td>
        <td align="center">Supports functions such as text_to_image, image_to_image, inpainting, ACGN external service, etc.</td>
        <td align="center">Support Chinese and English input</td>
        
<tr>

</table>
</div>




#### 👓 [Computer Vision Models](./modules#Image)
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_all.gif"  width = "530" height = "400" />
</div>


- Many thanks to CopyRight@[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN), [AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2)、[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), [Zhengxia Zou](https://github.com/jiupinjia/SkyAR)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) for the pre-trained models, you can try to train your models with them.


#### 🎤 [Natural Language Processing Models](./modules#Text)
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_all.gif"  width = "640" height = "240" />
</div>

- Many thanks to CopyRight@[ERNIE](https://github.com/PaddlePaddle/ERNIE)、[LAC](https://github.com/baidu/LAC)、[DDParser](https://github.com/baidu/DDParser)for the pre-trained models, you can try to train your models with them.



#### 🎧 [Speech Models](./modules#Audio)
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
<div align="center">
<table>
    <thead>
    </thead>
    <tbody>
        <tr>
            <th>Input Text </th>
            <th>Output Audio </th>
        </tr>
        <tr>
            <th>Life was like a box of chocolates, you never know what you're gonna get.</th>
            <th>
            <a href="https://paddlehub.bj.bcebos.com/resources/fastspeech_ljspeech-0.wav">
            <img src="./docs/imgs/Readme_Related/audio_icon.png" width=250 /></a><br>
            </th>
        </tr>
    </tbody>
</table>
</div>

- Many thanks to CopyRight@[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech) for the pre-trained models, you can try to train your models with PaddleSpeech.


### ⭐ Thanks for Your Star 
- All the above pre-trained models are all **open source and free**, and the number of models is continuously updated. Welcome **Star** to pay attention.
<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleHub/stargazers">
    <img src="./docs/imgs/Readme_Related/star_en.png"  width = "411" height = "100" /></a>  
</div>

<a name="Welcome_joinus"></a>

## 🍻Welcome to join PaddleHub technical group

- If you have any questions during the use of the model, you can join the official WeChat group to get more efficient questions and answers, and fully communicate with developers from all walks of life. We look forward to your joining.
<div align="center">
<img src="./docs/imgs/joinus.PNG"  width = "200" height = "200" />
</div> 

- please add WeChat above and send "Hub" to the robot, the robot will invite you to join the group automatically.

<a name="QuickStart"></a>
## ✈️QuickStart

#### 🚁The installation of required components.
```python
# install paddlepaddle with gpu
# !pip install --upgrade paddlepaddle-gpu

# or install paddlepaddle with cpu
!pip install --upgrade paddlepaddle

# install paddlehub
!pip install --upgrade paddlehub
```

#### 🛫The simplest cases of Chinese word segmentation.

```python
import paddlehub as hub

lac = hub.Module(name="lac")
test_text = ["今天是个好天气。"]

results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
print(results)
#{'word': ['今天', '是', '个', '好天气', '。'], 'tag': ['TIME', 'v', 'q', 'n', 'w']}
```
#### 🛰️The simplest command of deploying lac service.
</div>

```python
!hub serving start -m lac
```

- 📣More model description, please refer [Models List](./modules)

<a name="License"></a>
## 📚License
The release of this project is certified by the <a href="./LICENSE">Apache 2.0 license</a>.

<a name="Contribution"></a>
## 👨‍👨‍👧‍👦Contribution

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

We welcome you to contribute code to PaddleHub, and thank you for your feedback.

* Many thanks to [肖培楷](https://github.com/jm12138), Contributed to street scene cartoonization, portrait cartoonization, gesture key point recognition, sky replacement, depth estimation, portrait segmentation and other modules
* Many thanks to [Austendeng](https://github.com/Austendeng) for fixing the SequenceLabelReader
* Many thanks to [cclauss](https://github.com/cclauss) optimizing travis-ci check
* Many thanks to [奇想天外](http://www.cheerthink.com/)，Contributed a demo of mask detection
* Many thanks to [mhlwsk](https://github.com/mhlwsk)，Contributed the repair sequence annotation prediction demo
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp)，Contributed modules for viewing pictures and writing poems
* Many thanks to [zbp-xxxp](https://github.com/zbp-xxxp) and [七年期限](https://github.com/1084667371),Jointly contributed to the Mid-Autumn Festival Special Edition Module
* Many thanks to [livingbody](https://github.com/livingbody)，Contributed models for style transfer based on PaddleHub's capabilities and Mid-Autumn Festival WeChat Mini Program
* Many thanks to [BurrowsWang](https://github.com/BurrowsWang) for fixing Markdown table display problem
* Many thanks to [huqi](https://github.com/hu-qi) for fixing readme typo
* Many thanks to [parano](https://github.com/parano) [cqvu](https://github.com/cqvu) [deehrlic](https://github.com/deehrlic) for contributing this feature in PaddleHub
* Many thanks to [paopjian](https://github.com/paopjian) for correcting the wrong website address [#1424](https://github.com/PaddlePaddle/PaddleHub/issues/1424)
* Many thanks to [Wgm-Inspur](https://github.com/Wgm-Inspur) for correcting the demo errors in readme, and updating the RNN illustration in the text classification and sequence labeling demo
* Many thanks to [zl1271](https://github.com/zl1271) for fixing serving docs typo
* Many thanks to [AK391](https://github.com/AK391) for adding the webdemo of UGATIT and deoldify models in Hugging Face spaces
* Many thanks to [itegel](https://github.com/itegel) for fixing quick start docs typo
* Many thanks to [AK391](https://github.com/AK391) for adding the webdemo of Photo2Cartoon model in Hugging Face spaces
