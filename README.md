English | [简体中文](README_ch.md)

<p align="center">
 <img src="./docs/imgs/paddlehub_logo.jpg" align="middle">
<p align="center">
<div align="center">  
  <h3> <a href=#QuickStart> QuickStart </a> | <a href="https://paddlehub.readthedocs.io/en/release-v2.1"> Tutorial </a> | <a href="https://www.paddlepaddle.org.cn/hublist"> Models List </a> | <a href="https://www.paddlepaddle.org.cn/hub"> Demos </a> </h3>
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



## Introduction and Features
- **PaddleHub** aims to provide developers with rich, high-quality, and directly usable pre-trained models.
- **Abundant Pre-trained Models**: 300+ pre-trained models cover the 5 major categories, including Image, Text, Audio, Video, and Industrial application. All of them are free for download and offline usage.
- **No Need for Deep Learning Background**: you can use AI models quickly and enjoy the dividends of the artificial intelligence era.
- **Quick Model Prediction**: model prediction can be realized through a few lines of scripts to quickly experience the model effect.
- **Model As Service**: one-line command to build deep learning model API service deployment capabilities.
- **Easy-to-use Transfer Learning**: few lines of codes to complete the transfer-learning task such as image classification and text classification based on high quality pre-trained models.
- **Cross-platform**: support Linux, Windows, MacOS and other operating systems.

### Recent updates
- **2021.05.12:** Add an open-domain dialogue system, i.e., [plato-mini](https://www.paddlepaddle.org.cn/hubdetail?name=plato-mini&en_category=TextGeneration), to make it easy to build a chatbot in wechat with the help of the wechaty, [See Demo](https://github.com/KPatr1ck/paddlehub-wechaty-demo)
- **2021.04.27:** The v2.1.0 version is released. [1] Add supports for five new models, including two high-precision semantic segmentation models based on VOC dataset and three voice classification models. [2] Enforce the transfer learning capabilities for image semantic segmentation, text semantic matching and voice classification on related datasets. [3] Add the export function APIs for two kinds of model formats, i.,e, ONNX and PaddleInference. [4] Add the support for [BentoML](https://github.com/bentoml/BentoML/), which is a cloud native framework for serving deployment. Users can easily serve pre-trained models from PaddleHub by following the [Tutorial notebooks](https://github.com/PaddlePaddle/PaddleHub/blob/release/v2.1/demo/serving/bentoml/cloud-native-model-serving-with-bentoml.ipynb). Also, see this announcement and [Release note](https://github.com/bentoml/BentoML/releases/tag/v0.12.1) from BentoML. (Many thanks to @[parano](https://github.com/parano) @[cqvu](https://github.com/cqvu) @[deehrlic](https://github.com/deehrlic) for contributing this feature in PaddleHub). [5] The total number of pre-trained models reaches **【300】**.
- **2021.02.18:** The v2.0.0 version is released, making model development and debugging easier, and the finetune task is more flexible and easy to use.The ability to transfer learning for visual tasks is fully upgraded, supporting various tasks such as image classification, image coloring, and style transfer; Transformer models such as BERT, ERNIE, and RoBERTa are upgraded to dynamic graphs, supporting Fine-Tune capabilities for text classification and sequence labeling; Optimize the Serving capability, support multi-card prediction, automatic load balancing, and greatly improve performance; the new automatic data enhancement capability Auto Augment can efficiently search for data enhancement strategy combinations suitable for data sets. 61 new word vector models were added, including 51 Chinese models and 10 English models; add 4 image segmentation models, 2 depth models, 7 image generation models, and 3 text generation models, the total number of pre-trained models reaches **【274】**.
- [【more】](./docs/docs_en/release.md)




## Visualization Demo [[More]](./docs/docs_en/visualization.md)
### **Computer Vision (161 models)**
<div align="center">
<img src="./docs/imgs/Readme_Related/Image_all.gif"  width = "530" height = "400" />
</div>

- Many thanks to CopyRight@[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)、[PaddleGAN](https://github.com/PaddlePaddle/PaddleGAN)、[AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2)、[openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)、[Zhengxia Zou](https://github.com/jiupinjia/SkyAR)、[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) for the pre-trained models, you can try to train your models with them.


### **Natural Language Processing (129 models)**
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_all.gif"  width = "640" height = "240" />
</div>

- Many thanks to CopyRight@[ERNIE](https://github.com/PaddlePaddle/ERNIE)、[LAC](https://github.com/baidu/LAC)、[DDParser](https://github.com/baidu/DDParser)for the pre-trained models, you can try to train your models with them.



### Speech (3 models)
- TTS speech synthesis algorithm, multiple algorithms are available.
- Many thanks to CopyRight@[Parakeet](https://github.com/PaddlePaddle/Parakeet) for the pre-trained models, you can try to train your models with Parakeet.
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

### Video (8 models)
- Short video classification trained via large-scale video datasets, supports 3000+ tag types prediction for short Form Videos.
- Many thanks to CopyRight@[PaddleVideo](https://github.com/PaddlePaddle/PaddleVideo) for the pre-trained model, you can try to train your models with PaddleVideo.
- `Example: Input a short video of swimming, the algorithm can output the result of "swimming"`
<div align="center">
<img src="./docs/imgs/Readme_Related/Text_Video.gif"  width = "400" height = "400" />
</div>

## ===**Key Points**===
- All the above pre-trained models are all open source and free, and the number of models is continuously updated. Welcome **⭐Star⭐** to pay attention.
<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleHub/stargazers">
    <img src="./docs/imgs/Readme_Related/star_en.png"  width = "411" height = "100" /></a>  
</div>

<a name="Welcome_joinus"></a>

## Welcome to join PaddleHub technical group

If you have any questions during the use of the model, you can join the official WeChat group to get more efficient questions and answers, and fully communicate with developers from all walks of life. We look forward to your joining.
<div align="center">
<img src="https://user-images.githubusercontent.com/23690325/147116815-d99c8b54-716f-4009-a6cd-87f2621d248f.jpg"  width = "200" height = "200" />
</div>  
please add WeChat above and send "Hub" to the robot, the robot will invite you to join the group automatically.

<a name="QuickStart"></a>
## QuickStart

### The installation of required components.
```python
# install paddlepaddle with gpu
# !pip install --upgrade paddlepaddle-gpu

# or install paddlepaddle with cpu
!pip install --upgrade paddlepaddle

# install paddlehub
!pip install --upgrade paddlehub
```

### The simplest cases of Chinese word segmentation.

```python
import paddlehub as hub

lac = hub.Module(name="lac")
test_text = ["今天是个好天气。"]

results = lac.cut(text=test_text, use_gpu=False, batch_size=1, return_tag=True)
print(results)
#{'word': ['今天', '是', '个', '好天气', '。'], 'tag': ['TIME', 'v', 'q', 'n', 'w']}
```
### The simplest command of deploying lac service.
</div>

```python
!hub serving start -m lac
```

More model description, please refer [Models List](https://www.paddlepaddle.org.cn/hublist)

More API for transfer learning, please refer [Tutorial](https://paddlehub.readthedocs.io/en/release-v2.1/transfer_learning_index.html)

<a name="License"></a>
## License
The release of this project is certified by the <a href="./LICENSE">Apache 2.0 license</a>.

<a name="Contribution"></a>
## Contribution

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
