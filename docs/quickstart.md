# 快速体验

安装PaddleHub成功后，执行命令[hub run](https://github.com/PaddlePaddle/PaddleHub/wiki/PaddleHub%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%B7%A5%E5%85%B7#run)，可以快速体验PaddleHub无需代码、一键预测的命令行功能，如下三个示例：

使用[词法分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=LexicalAnalysis)模型LAC进行分词
```shell
$ hub run lac --input_text "今天是个好日子"
[{'word': ['今天', '是', '个', '好日子'], 'tag': ['TIME', 'v', 'q', 'n']}]
```

使用[情感分析](http://www.paddlepaddle.org.cn/hub?filter=category&value=SentimentAnalysis)模型Senta对句子进行情感预测
```shell
$ hub run senta_bilstm --input_text "今天天气真好"
{'text': '今天天气真好', 'sentiment_label': 1, 'sentiment_key': 'positive', 'positive_probs': 0.9798, 'negative_probs': 0.0202}]
```

使用[目标检测](http://www.paddlepaddle.org.cn/hub?filter=category&value=ObjectDetection)模型Ultra-Light-Fast-Generic-Face-Detector-1MB对图片进行人脸识别
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ultra_light_fast_generic_face_detector_1mb_640 --input_path test_image.jpg
```
![人脸识别结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.5/docs/imgs/face_detection_result.png)

使用[图像分割](https://www.paddlepaddle.org.cn/hub?filter=en_category&value=ImageSegmentation)模型ace2p对图片进行tu
```shell
$ wget https://paddlehub.bj.bcebos.com/resources/test_image.jpg
$ hub run ace2p --input_path test_image.jpg
```
![图像分割结果](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v1.5/docs/imgs/img_seg_result.png)

除了上述三类模型外，PaddleHub还发布了图像分类、语义模型、视频分类、图像生成、图像分割、文本审核、关键点检测等业界主流模型，更多PaddleHub已经发布的模型，请前往 https://www.paddlepaddle.org.cn/hub 查看
