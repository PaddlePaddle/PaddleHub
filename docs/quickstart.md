# 快速体验

安装PaddleHub成功后，执行命令[hub run](tutorial/cmdintro.md)，可以快速体验PaddleHub无需代码、一键预测的命令行功能，如下几个示例：

* 使用[文字识别](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=TextRecognition)轻量级中文OCR模型chinese_ocr_db_crnn_mobile即可一键快速识别图片中的文字。
```shell
$ wget https://paddlehub.bj.bcebos.com/model/image/ocr/test_ocr.jpg
$ hub run chinese_ocr_db_crnn_mobile --input_path test_ocr.jpg --visualization=True
```

预测结果图片保存在当前运行路径下ocr_result文件夹中，如下图所示。

![](./imgs/ocr_res.jpg)


* 使用[词法分析](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=LexicalAnalysis)模型LAC进行分词
```shell
$ hub run lac --input_text "现在，慕尼黑再保险公司不仅是此类行动的倡议者，更是将其大量气候数据整合进保险产品中，并与公众共享大量天气信息，参与到新能源领域的保障中。"
[{
    'word': ['现在', '，', '慕尼黑再保险公司', '不仅', '是', '此类', '行动', '的', '倡议者', '，', '更是', '将', '其', '大量', '气候', '数据', '整合', '进', '保险', '产品', '中', '，', '并', '与', '公众', '共享', '大量', '天气', '信息', '，', '参与', '到', '新能源', '领域', '的', '保障', '中', '。'],
    'tag':  ['TIME', 'w', 'ORG', 'c', 'v', 'r', 'n', 'u', 'n', 'w', 'd', 'p', 'r', 'a', 'n', 'n', 'v', 'v', 'n', 'n', 'f', 'w', 'c', 'p', 'n', 'v', 'a', 'n', 'n', 'w', 'v', 'v', 'n', 'n', 'u', 'vn', 'f', 'w']
}]
```

* 使用[情感分析](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=SentimentAnalysis)模型Senta对句子进行情感预测
```shell
$ hub run senta_bilstm --input_text "今天天气真好"
{'text': '今天天气真好', 'sentiment_label': 1, 'sentiment_key': 'positive', 'positive_probs': 0.9798, 'negative_probs': 0.0202}]
```

* 使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型Ultra-Light-Fast-Generic-Face-Detector-1MB对图片进行人脸识别
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ultra_light_fast_generic_face_detector_1mb_640 --input_path test_image.jpg
```

![](./imgs/face_detection_result.jpeg)


* 使用[图像分割](https://www.paddlepaddle.org.cn/hublist?filter=en_category&value=ImageSegmentation)模型进行人像扣图和人体部件识别

```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ace2p --input_path test_image.jpg
$ hub run deeplabv3p_xception65_humanseg --input_path test_image.jpg
```

![人像抠图](./imgs/img_seg_result.jpeg) ![人体部件识别](./imgs/humanseg_test_res.png)


除了上述几类模型外，PaddleHub还发布了图像分类、语义模型、视频分类、图像生成、图像分割、文本审核、关键点检测等业界主流模型，更多PaddleHub已经发布的模型，请前往 [PaddleHub官网](https://www.paddlepaddle.org.cn/hub) 查看
