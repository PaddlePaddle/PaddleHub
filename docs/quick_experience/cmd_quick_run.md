# 通过命令行调用方式使用PaddleHub

本页面的代码/命令可在[AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/643120)上在线运行，类似notebook的环境，只需通过浏览器即可访问，无需准备环境，非常方便开发者快速体验。

PaddleHub在设计时，为模型的管理和使用提供了命令行工具，也提供了通过命令行调用PaddleHub模型完成预测的方式。比如，前面章节中人像分割和文本分词的任务也可以通过命令行调用的方式实现。

### 体验前请提前安装好PaddleHub


```shell
# 安装最新版本，使用清华源更稳定、更迅速
$ pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 人像扣图


```shell
# 下载待测试图片
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
# 通过命令行方式实现人像扣图任务
$ hub run deeplabv3p_xception65_humanseg --input_path test_image.jpg --visualization=True --output_dir="humanseg_output"
```

    --2020-07-22 12:19:52--  https://paddlehub.bj.bcebos.com/resources/test_image.jpg
    Resolving paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)... 182.61.200.195, 182.61.200.229
    Connecting to paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 967120 (944K) [image/jpeg]
    Saving to: ‘test_image.jpg’

    test_image.jpg      100%[===================>] 944.45K  6.13MB/s    in 0.2s  

    2020-07-22 12:19:53 (6.13 MB/s) - ‘test_image.jpg’ saved [967120/967120]

    [{'save_path': 'humanseg_output/test_image.png', 'data': array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32)}]



![png](../imgs/humanseg_test_res.png)


### 中文分词


```shell
#通过命令行方式实现文本分词任务
$ hub run lac --input_text "今天是个好日子"
```

    Install Module lac
    Downloading lac
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpjcskpj8x/lac
    [==================================================] 100.00%
    Successfully installed lac-2.1.1
    [{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]


上面的命令中包含四个部分，分别是：
- hub 表示PaddleHub的命令。
- run 调用run执行模型的预测。
- deeplabv3p_xception65_humanseg、lac 表示要调用的算法模型。
- --input_path/--input_text  表示模型的输入数据，图像和文本的输入方式不同。

另外，命令行中`visualization=True`表示将结果可视化输出，`output_dir="humanseg_output"`表示预测结果的保存目录，可以到该路径下查看输出的图片。

再看一个文字识别和一个口罩检测的例子。

### OCR文字识别


```shell
# 下载待测试的图片
$ wget https://paddlehub.bj.bcebos.com/model/image/ocr/test_ocr.jpg

# 该Module依赖于第三方库shapely和pyclipper，需提前安装
$ pip install shapely
$ pip install pyclipper

# 通过命令行方式实现文字识别任务
$ hub run chinese_ocr_db_crnn_mobile --input_path test_ocr.jpg --visualization=True --output_dir='ocr_result'
```

    --2020-07-22 15:00:50--  https://paddlehub.bj.bcebos.com/model/image/ocr/test_ocr.jpg
    Resolving paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)... 182.61.200.195, 182.61.200.229
    Connecting to paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)|182.61.200.195|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 48680 (48K) [image/jpeg]
    Saving to: ‘test_ocr.jpg’

    test_ocr.jpg        100%[===================>]  47.54K  --.-KB/s    in 0.02s  

    2020-07-22 15:00:51 (2.88 MB/s) - ‘test_ocr.jpg’ saved [48680/48680]

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Requirement already satisfied: shapely in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.7.0)
    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Requirement already satisfied: pyclipper in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (1.2.0)
    [{'save_path': 'ocr_result/ndarray_1595401261.294494.jpg', 'data': [{'text': '纯臻营养护发素', 'confidence': 0.9438689351081848, 'text_box_position': [[24, 36], [304, 34], [304, 72], [24, 74]]}, {'text': '产品信息/参数', 'confidence': 0.9843138456344604, 'text_box_position': [[24, 80], [172, 80], [172, 104], [24, 104]]}, {'text': '（45元/每公斤，100公斤起订）', 'confidence': 0.9210420250892639, 'text_box_position': [[24, 109], [333, 109], [333, 136], [24, 136]]}, {'text': '每瓶22元，1000瓶起订）', 'confidence': 0.9685984253883362, 'text_box_position': [[22, 139], [283, 139], [283, 166], [22, 166]]}, {'text': '【品牌】', 'confidence': 0.9527574181556702, 'text_box_position': [[22, 174], [85, 174], [85, 198], [22, 198]]}, {'text': '：代加工方式/OEMODM', 'confidence': 0.9442129135131836, 'text_box_position': [[90, 176], [301, 176], [301, 196], [90, 196]]}, {'text': '【品名】', 'confidence': 0.8793742060661316, 'text_box_position': [[23, 205], [85, 205], [85, 229], [23, 229]]}, {'text': '：纯臻营养护发素', 'confidence': 0.9230973124504089, 'text_box_position': [[95, 204], [235, 206], [235, 229], [95, 227]]}, {'text': '【产品编号】', 'confidence': 0.9311650395393372, 'text_box_position': [[24, 238], [120, 238], [120, 260], [24, 260]]}, {'text': 'J：YM-X-3011', 'confidence': 0.8866629004478455, 'text_box_position': [[110, 239], [239, 239], [239, 256], [110, 256]]}, {'text': 'ODMOEM', 'confidence': 0.9916308522224426, 'text_box_position': [[414, 233], [430, 233], [430, 304], [414, 304]]}, {'text': '【净含量】：220ml', 'confidence': 0.8709315657615662, 'text_box_position': [[23, 268], [181, 268], [181, 292], [23, 292]]}, {'text': '【适用人群】', 'confidence': 0.9589888453483582, 'text_box_position': [[24, 301], [118, 301], [118, 321], [24, 321]]}, {'text': '：适合所有肤质', 'confidence': 0.935418963432312, 'text_box_position': [[131, 300], [254, 300], [254, 323], [131, 323]]}, {'text': '【主要成分】', 'confidence': 0.9366627335548401, 'text_box_position': [[24, 332], [117, 332], [117, 353], [24, 353]]}, {'text': '鲸蜡硬脂醇', 'confidence': 0.9033458828926086, 'text_box_position': [[138, 331], [235, 331], [235, 351], [138, 351]]}, {'text': '燕麦B-葡聚', 'confidence': 0.8497812747955322, 'text_box_position': [[248, 332], [345, 332], [345, 352], [248, 352]]}, {'text': '椰油酰胺丙基甜菜碱、', 'confidence': 0.8935506939888, 'text_box_position': [[54, 363], [232, 363], [232, 383], [54, 383]]}, {'text': '糖、', 'confidence': 0.8750994205474854, 'text_box_position': [[25, 364], [62, 364], [62, 383], [25, 383]]}, {'text': '泛酯', 'confidence': 0.5581164956092834, 'text_box_position': [[244, 363], [281, 363], [281, 382], [244, 382]]}, {'text': '（成品包材）', 'confidence': 0.9566792845726013, 'text_box_position': [[368, 367], [475, 367], [475, 388], [368, 388]]}, {'text': '【主要功能】', 'confidence': 0.9493741393089294, 'text_box_position': [[24, 395], [119, 395], [119, 416], [24, 416]]}, {'text': '：可紧致头发磷层', 'confidence': 0.9692543745040894, 'text_box_position': [[128, 397], [273, 397], [273, 414], [128, 414]]}, {'text': '美，从而达到', 'confidence': 0.8662520051002502, 'text_box_position': [[265, 395], [361, 395], [361, 415], [265, 415]]}, {'text': '即时持久改善头发光泽的效果，给干燥的头', 'confidence': 0.9690631031990051, 'text_box_position': [[25, 425], [372, 425], [372, 448], [25, 448]]}, {'text': '发足够的滋养', 'confidence': 0.8946213126182556, 'text_box_position': [[26, 457], [136, 457], [136, 477], [26, 477]]}]}]



```shell
# 查看预测结果
```


![png](../imgs/ocr_res.jpg)


### 口罩检测


```shell
# 下载待测试的图片
$ wget https://paddlehub.bj.bcebos.com/resources/test_mask_detection.jpg

# 通过命令行方式实现文字识别任务
$ hub run pyramidbox_lite_mobile_mask --input_path test_mask_detection.jpg --visualization=True --output_dir='detection_result'
```

    --2020-07-22 15:08:11--  https://paddlehub.bj.bcebos.com/resources/test_mask_detection.jpg
    Resolving paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)... 182.61.200.229, 182.61.200.195
    Connecting to paddlehub.bj.bcebos.com (paddlehub.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 299133 (292K) [image/jpeg]
    Saving to: ‘test_mask_detection.jpg’

    test_mask_detection 100%[===================>] 292.12K  --.-KB/s    in 0.06s  

    2020-07-22 15:08:11 (4.55 MB/s) - ‘test_mask_detection.jpg’ saved [299133/299133]

    Install Module pyramidbox_lite_mobile_mask
    Downloading pyramidbox_lite_mobile_mask
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmp8oes9jid/pyramidbox_lite_mobile_mask
    [==================================================] 100.00%
    Successfully installed pyramidbox_lite_mobile_mask-1.3.0
    Downloading pyramidbox_lite_mobile
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpvhjhlr10/pyramidbox_lite_mobile
    [==================================================] 100.00%
    [{'data': [{'label': 'MASK', 'confidence': 0.9992434978485107, 'top': 181, 'bottom': 440, 'left': 457, 'right': 654}, {'label': 'MASK', 'confidence': 0.9224318265914917, 'top': 340, 'bottom': 578, 'left': 945, 'right': 1125}, {'label': 'NO MASK', 'confidence': 0.9996706247329712, 'top': 292, 'bottom': 500, 'left': 1166, 'right': 1323}], 'path': 'test_mask_detection.jpg'}]



```shell
# 查看预测结果
```


![png](../imgs/test_mask_detection_result.jpg)


### PaddleHub命令行工具简介

PaddleHub的命令行工具在开发时借鉴了Anaconda和PIP等软件包管理的理念，可以方便快捷的完成模型的搜索、下载、安装、升级、预测等功能。 下面概要介绍一下PaddleHub支持的12个命令，详细介绍可查看[命令行参考](../tutorial/cmdintro.md)章节。：
* install：用于将Module安装到本地，默认安装在{HUB_HOME}/.paddlehub/modules目录下；
* uninstall：卸载本地Module；
* show：用于查看本地已安装Module的属性或者指定目录下确定的Module的属性，包括其名字、版本、描述、作者等信息；
* download：用于下载百度飞桨PaddleHub提供的Module；
* search：通过关键字在服务端检索匹配的Module，当想要查找某个特定模型的Module时，使用search命令可以快速得到结果，例如hub search ssd命令，会查找所有包含了ssd字样的Module，命令支持正则表达式，例如hub search ^s.\*搜索所有以s开头的资源；
* list：列出本地已经安装的Module；
* run：用于执行Module的预测；
* version：显示PaddleHub版本信息；
* help：显示帮助信息；
* clear：PaddleHub在使用过程中会产生一些缓存数据，这部分数据默认存放在${HUB_HOME}/.paddlehub/cache目录下，用户可以通过clear命令来清空缓存；
* autofinetune：用于自动调整Fine-tune任务的超参数，具体使用详情参考[PaddleHub AutoDL Finetuner](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.5/docs/tutorial/autofinetune.md)使用教程；
* config：用于查看和设置Paddlehub相关设置，包括对server地址、日志级别的设置；
* serving：用于一键部署Module预测服务，详细用法见[PaddleHub Serving一键服务部署](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.5/docs/tutorial/serving.md)。

## 小结
PaddleHub的产品理念是模型即软件，通过Python API或命令行实现模型调用，可快速体验或集成飞桨特色预训练模型。
此外，当用户想用少量数据来优化预训练模型时，PaddleHub也支持迁移学习，通过Fine-tune API，内置多种优化策略，只需少量代码即可完成预训练模型的Fine-tuning。具体可通过后面迁移学习的章节了解。
>值得注意的是，不是所有的Module都支持通过命令行预测 (例如BERT/ERNIE Transformer类模型，一般需要搭配任务进行Fine-tune)， 也不是所有的Module都可用于Fine-tune（例如一般不建议用户使用词法分析LAC模型Fine-tune）。建议提前阅读[预训练模型的介绍文档](https://www.paddlepaddle.org.cn/hublist)了解使用场景。
