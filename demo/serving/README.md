# PaddleHub-Serving
## 1. 简介
利用PaddleHub-Serving可以完成模型服务化部署，主要包括利用Bert as Service实现embedding服务化，利用预测模型实现预测服务化。

## 2. Bert as Service
* [Bert as Service介绍与示例](bert_service)

该示例展示了利用Bert as Service进行远程embedding服务化部署和在线预测，获取文本embedding结果。

## 3. Serving
模型预测服务化有以下示例：  

* [图像分类-基于vgg11_imagent](./Classification_vgg11_imagenet)  

该示例展示了利用vgg11_imagent完成图像分类服务化部署和在线预测，获取图像分类结果。

* [图像生成-基于stgan_celeba](./GAN_stgan_celeba)  

该示例展示了利用stgan_celeba生成图像服务化部署和在线预测，获取指定风格的生成图像。

* [英文词法分析-基于lm_lstm](./Language_Model_lm_lstm)

该示例展示了利用lm_lstm完成英文语法分析服务化部署和在线预测，获取文本的流利程度。

* [中文词法分析-基于lac](./Lexical_Analysis_lac)

该示例展示了利用lac完成中文文本分词服务化部署和在线预测，获取文本的分词结果，并可通过用户自定义词典干预分词结果。

* [目标检测-基于yolov3_coco2017](./Object_Detection_yolov3_coco2017)  

该示例展示了利用yolov3_coco2017完成目标检测服务化部署和在线预测，获取检测结果和覆盖识别框的图片。

* [中文语义分析-基于simnet_bow](./Semantic_Model_simnet_bow)

该示例展示了利用simnet_bow完成中文文本相似度检测服务化部署和在线预测，获取文本的相似程度。  

* [图像分割-基于deeplabv3p_xception65_humanseg](./Semantic_Segmentation_deeplabv3p_xception65_humanseg)

该示例展示了利用deeplabv3p_xception65_humanseg完成图像分割服务化部署和在线预测，获取识别结果和分割后的图像。

* [中文情感分析-基于senta_lstm](./Sentiment_Analysis_senta_lstm)

该示例展示了利用senta_lstm完成中文文本情感分析服务化部署和在线预测，获取文本的情感分析结果。
