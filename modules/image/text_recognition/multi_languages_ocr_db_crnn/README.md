# multi_languages_ocr_db_crnn

|模型名称|multi_languages_ocr_db_crnn|
| :--- | :---: |
|类别|图像-文字识别|
|网络|Differentiable Binarization+RCNN|
|数据集|icdar2015数据集|
|是否支持Fine-tuning|否|
|最新更新日期|2021-11-24|
|数据指标|-|


## 一、模型基本信息

- ### 应用效果展示
  - 样例结果示例：
<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133097562-d8c9abd1-6c70-4d93-809f-fa4735764836.png"  width = "600" hspace='10'/> <br />
</p>

- ### 模型介绍

  - multi_languages_ocr_db_crnn Module用于识别图片当中的文字。其基于PaddleOCR模块，检测得到文本框，识别文本框中的文字,再对检测文本框进行角度分类。最终检测算法采用DB(Differentiable Binarization)，而识别文字算法则采用CRNN（Convolutional Recurrent Neural Network）即卷积递归神经网络。
    该Module不仅提供了通用场景下的中英文模型，也提供了[80个语言](#语种缩写)的小语种模型。


<p align="center">
<img src="https://user-images.githubusercontent.com/76040149/133098254-7c642826-d6d7-4dd0-986e-371622337867.png" width = "300" height = "450"  hspace='10'/> <br />
</p>

  - 更多详情参考：
    - [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf)
    - [An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition](https://arxiv.org/pdf/1507.05717.pdf)



## 二、安装

- ### 1、环境依赖  

  - paddlepaddle >= 2.0.2  

  - paddlehub >= 2.0.0   | [如何安装paddlehub](../../../../docs/docs_ch/get_start/installation.rst)

- ### 2、安装

  - ```shell
    $ hub install multi_languages_ocr_db_crnn
    ```
  - 如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
 | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)



## 三、模型API预测

- ### 1、命令行预测

  - ```shell
    $ hub run multi_languages_ocr_db_crnn --input_path "/PATH/TO/IMAGE"
    $ hub run multi_languages_ocr_db_crnn --input_path "/PATH/TO/IMAGE" --lang "ch"  --det True --rec True --use_angle_cls True  --box_thresh 0.7 --angle_classification_thresh 0.8 --visualization True
    ```
  - 通过命令行方式实现文字识别模型的调用，更多请见 [PaddleHub命令行指令](../../../../docs/docs_ch/tutorial/cmd_usage.rst)

- ### 2、预测代码示例

  - ```python
    import paddlehub as hub
    import cv2

    ocr = hub.Module(name="multi_languages_ocr_db_crnn", lang='en', enable_mkldnn=True)       # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread('/PATH/TO/IMAGE')])

    # or
    # result = ocr.recognize_text(paths=['/PATH/TO/IMAGE'])
    ```
  - multi_languages_ocr_db_crnn目前支持80个语种，可以通过修改lang参数进行切换，对于英文模型，指定lang=en，具体支持的[语种](#语种缩写)可查看表格。

- ### 3、API

  - ```python
    def __init__(self,
                 lang="ch",
                 det=True, rec=True,
                 use_angle_cls=False,
                 enable_mkldnn=False,  
                 use_gpu=False,
                 box_thresh=0.6,
                 angle_classification_thresh=0.9)
    ```

    - 构造MultiLangOCR对象

    - **参数**
      - lang(str): 多语言模型选择。默认为中文模型，即lang="ch"。
      - det(bool): 是否开启文字检测。默认为True。
      - rec(bool): 是否开启文字识别。默认为True。
      - use_angle_cls(bool): 是否开启方向分类, 用于设置使用方向分类器识别180度旋转文字。默认为False。
      - enable_mkldnn(bool): 是否开启mkldnn加速CPU计算。该参数仅在CPU运行下设置有效。默认为False。
      - use\_gpu (bool): 是否使用 GPU；**若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量**
      - box\_thresh (float): 检测文本框置信度的阈值；
      - angle_classification_thresh(float): 文本方向分类置信度的阈值


  - ```python
    def recognize_text(images=[],
                       paths=[],
                       output_dir='ocr_result',
                       visualization=False)
    ```

    - 预测API，检测输入图片中的所有文本的位置和识别文本结果。

    - **参数**

      - paths (list\[str\]): 图片的路径；
      - images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
      - output\_dir (str): 图片的保存路径，默认设为 ocr\_result；
      - visualization (bool): 是否将识别结果保存为图片文件, 仅有检测开启时有效, 默认为False；

    - **返回**

      - res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
        - data (list\[dict\]): 识别文本结果，列表中每一个元素为 dict，各字段为：
          - text(str): 识别得到的文本
          - confidence(float): 识别文本结果置信度
          - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标，如果无识别结果则data为\[\]
          - orientation(str): 分类的方向，仅在只有方向分类开启时输出
          - score(float): 分类的得分，仅在只有方向分类开启时输出
        - save_path (str, optional): 识别结果的保存路径，如不保存图片则save_path为''


## 四、服务部署

- PaddleHub Serving 可以部署一个目标检测的在线服务。

- ### 第一步：启动PaddleHub Serving

  - 运行启动命令：
  - ```shell
    $ hub serving start -m multi_languages_ocr_db_crnn
    ```

  - 这样就完成了一个目标检测的服务化API的部署，默认端口号为8866。

  - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

  - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

  - ```python
    import requests
    import json
    import cv2
    import base64

    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    # 发送HTTP请求
    data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
    headers = {"Content-type": "application/json"}
    url = "http://127.0.0.1:8866/predict/multi_languages_ocr_db_crnn"
    r = requests.post(url=url, headers=headers, data=json.dumps(data))

    # 打印预测结果
    print(r.json()["results"])
    ```

<a name="语种缩写"></a>
## 五、支持语种及缩写

| 语种 | 描述 | 缩写 | | 语种 | 描述 | 缩写 |
| --- | --- | --- | ---|--- | --- | --- |
|中文|chinese and english|ch| |保加利亚文|Bulgarian |bg|
|英文|english|en| |乌克兰文|Ukranian|uk|
|法文|french|fr| |白俄罗斯文|Belarusian|be|
|德文|german|german| |泰卢固文|Telugu |te|
|日文|japan|japan| | 阿巴扎文 | Abaza        | abq  |
|韩文|korean|korean| |泰米尔文|Tamil |ta|
|中文繁体|chinese traditional |chinese_cht| |南非荷兰文 |Afrikaans |af|
|意大利文| Italian |it| |阿塞拜疆文 |Azerbaijani    |az|
|西班牙文|Spanish |es| |波斯尼亚文|Bosnian|bs|
|葡萄牙文| Portuguese|pt| |捷克文|Czech|cs|
|俄罗斯文|Russia|ru| |威尔士文 |Welsh |cy|
|阿拉伯文|Arabic|ar| |丹麦文 |Danish|da|
|印地文|Hindi|hi| |爱沙尼亚文 |Estonian |et|
|维吾尔|Uyghur|ug| |爱尔兰文 |Irish |ga|
|波斯文|Persian|fa| |克罗地亚文|Croatian |hr|
|乌尔都文|Urdu|ur| |匈牙利文|Hungarian |hu|
|塞尔维亚文（latin)| Serbian(latin) |rs_latin| |印尼文|Indonesian|id|
|欧西坦文|Occitan |oc| |冰岛文 |Icelandic|is|
|马拉地文|Marathi|mr| |库尔德文 |Kurdish|ku|
|尼泊尔文|Nepali|ne| |立陶宛文|Lithuanian |lt|
|塞尔维亚文（cyrillic)|Serbian(cyrillic)|rs_cyrillic| |拉脱维亚文 |Latvian |lv|
|毛利文|Maori|mi| | 达尔瓦文|Dargwa |dar|
|马来文 |Malay|ms| | 因古什文|Ingush |inh|
|马耳他文 |Maltese |mt| | 拉克文|Lak |lbe|
|荷兰文 |Dutch |nl| | 莱兹甘文|Lezghian |lez|
|挪威文 |Norwegian |no| |塔巴萨兰文 |Tabassaran |tab|
|波兰文|Polish |pl| | 比尔哈文|Bihari |bh|
| 罗马尼亚文|Romanian |ro| | 迈蒂利文|Maithili |mai|
| 斯洛伐克文|Slovak |sk| | 昂加文|Angika |ang|
| 斯洛文尼亚文|Slovenian |sl| | 孟加拉文|Bhojpuri |bho|
| 阿尔巴尼亚文|Albanian |sq| | 摩揭陀文 |Magahi |mah|
| 瑞典文|Swedish |sv| | 那格浦尔文|Nagpur |sck|
| 西瓦希里文|Swahili |sw| | 尼瓦尔文|Newari |new|
| 塔加洛文|Tagalog |tl| | 保加利亚文 |Goan Konkani|gom|
| 土耳其文|Turkish |tr| | 沙特阿拉伯文|Saudi Arabia|sa|
| 乌兹别克文|Uzbek |uz| | 阿瓦尔文|Avar |ava|
| 越南文|Vietnamese |vi| | 阿瓦尔文|Avar |ava|
| 蒙古文|Mongolian |mn| | 阿迪赫文|Adyghe |ady|

## 五、更新历史

* 1.0.0

  初始发布
  - ```shell
    $ hub install multi_languages_ocr_db_crnn==1.0.0
    ```
