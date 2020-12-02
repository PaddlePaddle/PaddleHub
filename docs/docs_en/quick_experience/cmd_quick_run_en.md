# Using PaddleHub through Command Line Execution

The codes/commands on this page can run online on [AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/643120). It is similar to the notebook environment, which can be accessed through a browser, without environment preparation. This is a quick and easy experiences for developers.

PaddleHub is designed to provide the command line tool for model management and usage. It also provides a method of completing predictions by executing the PaddleHub model in the command line. For example, the tasks of portrait segmentation and text word segmentation in the previous sections can also be implemented through command line execution.

### Before experience, install the PaddleHub.

```shell
# Lastest version
$ pip install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Portrait Cutout

```shell
# Download picture
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
# Inference with command line
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

![png](../../imgs/humanseg_test_res.png)

### Chinese word segmentation

```shell
# Inference with command line
$ hub run lac --input_text "今天是个好日子"
```

    Install Module lac
    Downloading lac
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpjcskpj8x/lac
    [==================================================] 100.00%
    Successfully installed lac-2.1.1
    [{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]

The above command contains four parts:

- Hub: indicates the PaddleHub command.
- Run: invokes run to execute the model prediction.
- deeplabv3p\_xception65\_humanseg and lac: Indicate the algorithm model to be executed.
- --input\_path/-input\_text: Indicates the input data for the model, with different input methods for images and text.

In addition, the command line `visualization=True` indicates the visual output of the results, and `output_dir="humanseg_output"` indicates the directory where the prediction results are saved. You can access this path to view the output images.

Let's look at an example of OCR and mask detection.

### OCR

```shell
# Download picture
$ wget https://paddlehub.bj.bcebos.com/model/image/ocr/test_ocr.jpg

# requirements
$ pip install shapely
$ pip install pyclipper

# Inference with command line
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
# check the result
```

![png](../../imgs/ocr_res.jpg)

### Mask Detection

```shell
# Download picture
$ wget https://paddlehub.bj.bcebos.com/resources/test_mask_detection.jpg

# Inference with command line
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
# check the result
```

![png](../../imgs/test_mask_detection_result.jpg)

### This is the introduction to the PaddleHub command line tool.

The command line tool of PaddleHub is developed with introducing package management concepts such as Anaconda and PIP. It can be used to search, download, install, upgrade, and predict models quickly and conveniently. The following is an overview of the 12 commands supported by PaddleHub. For details, see  Command Line Reference :

* install: Installs the module locally, by default, in {HUB\_HOME}/.paddlehub/modules directory.
* uninstall: Uninstalls the local module.
* show: Views the properties of locally installed module or the properties of a module identified in the specified directory, including its name, version, description, author and other information.
* download: Downloads the module provided by PaddleHub of Baidu.
* search: Searches for matching Modules on the server by keywords. When you want to find a Module of a specific model, run search command to get fast results. For example, hub search ssd: the command runs to search all Modules that contain ssd. The command supports regular expressions, for example, hub search \^s.\*: the command runs to search all resources beginning with s.
* list: Lists locally installed module.
* run: Executes the predictions of the module.
* version: displays PaddleHub version information.
* help: displays help information.
* clear: PaddleHub generates some cached data during operation, which is stored in ${HUB\_HOME}/.paddlehub/cache by default. Users can clear the cache by running the clear command.
* autofinetune: Automatically adjusts the hyper-parameters of Fine-tune tasks. For details, see [PaddleHub AutoDL Finetuner](https://github.com/PaddlePaddle/PaddleHub/blob/release/v1.5/docs/tutorial/autofinetune.md) Tutorial.
* config: Views and sets Paddlehub-related settings, including settings of server address and log level.
* serving: Deploys Module Prediction Services in one key. For details, see PaddleHub Serving One-Key Service Deployment.

## Summary

According to the PaddleHub product concept, the model is the software. It is executed through Python API or command line to allow you to quickly experience or integrate the pre-training models with the Paddle characteristics. In addition, when users want to optimize pre-training models with small amounts of data, PaddleHub also supports migration learning. Through the Fine-tune API and a variety of built-in optimization strategies, only a small number of codes is needed to complete the fine-tuning of pre-training models. You can learn more about this later in the chapter on Transfer Learning.

> It should be noted that not all modules support prediction through command line (for example, BERT/ERNIE Transformer class models, which generally require fine-tuning with tasks), and not all modules can be used for fine-tuning (for example, it is not recommended to use the fine-tune of  lexical analysis LAC models). It is recommended to read the Introduction to the pre-training model to understand the usage scenarios.
