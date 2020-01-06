# PaddleHub Serving
## 简介
### 背景
使用PaddleHub能够完成预训练模型的管理和一键预测，但开发者还经常面临将模型部署上线以对外提供服务的需求，而利用PaddleHub Serving可便捷的将模型部署上线，开发者只需要关注如何处理输入数据和输出结果即可。
### 主要功能
PaddleHub Serving是基于PaddleHub的一键模型服务部署工具，能够通过简单的Hub命令行工具轻松启动一个模型预测在线服务。

PaddleHub Serving主要包括利用Bert Service实现embedding服务化，以及利用预测模型实现预训练模型预测服务化两大功能，未来还将支持开发者使用PaddleHub Fine-tune API的模型服务化。

## Bert Service
Bert Service是基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)框架的快速部署模型远程计算服务方案，可将embedding过程通过调用API接口的方式实现，减少了对机器资源的依赖。使用PaddleHub可在服务器上一键部署`Bert Service`服务，在另外的普通机器上通过客户端接口即可轻松的获取文本对应的embedding数据。

关于Bert Service的具体信息和demo请参见[Bert Service](../../tutorial/bert_service.md)

该示例展示了利用Bert Service进行远程embedding服务化部署和在线预测，并获取文本embedding结果。

##  一键服务部署
一键服务部署是基于PaddleHub的预训练模型快速部署的服务化方案，能够将模型预测以API接口的方式实现。

关于一键服务部署的具体信息请参见[PaqddleHub Serving](../../tutorial/PaddleHub_serving.md)

一键服务部署包括以下示例：  

* [图像分类-基于vgg11_imagent](module_serving/classification_vgg11_imagenet)  

&emsp;&emsp;该示例展示了利用vgg11_imagent完成图像分类服务化部署和在线预测，获取图像分类结果。

* [图像生成-基于stgan_celeba](module_serving/GAN_stgan_celeba)  

&emsp;&emsp;该示例展示了利用stgan_celeba生成图像服务化部署和在线预测，获取指定风格的生成图像。

* [文本审核-基于porn_detection_lstm](module_serving/text_censorship_porn_detection_lstm)  

&emsp;&emsp;该示例展示了利用porn_detection_lstm完成中文文本黄色敏感信息鉴定的服务化部署和在线预测，获取文本是否敏感及其置信度。

* [中文词法分析-基于lac](module_serving/lexical_analysis_lac)

&emsp;&emsp;该示例展示了利用lac完成中文文本分词服务化部署和在线预测，获取文本的分词结果，并可通过用户自定义词典干预分词结果。

* [目标检测-基于yolov3_darknet53_coco2017](module_serving/object_detection_yolov3_darknet53_coco2017)  

&emsp;&emsp;该示例展示了利用yolov3_darknet53_coco2017完成目标检测服务化部署和在线预测，获取检测结果和覆盖识别框的图片。

* [中文语义分析-基于simnet_bow](module_serving/semantic_model_simnet_bow)

&emsp;&emsp;该示例展示了利用simnet_bow完成中文文本相似度检测服务化部署和在线预测，获取文本的相似程度。  

* [图像分割-基于deeplabv3p_xception65_humanseg](module_serving/semantic_segmentation_deeplabv3p_xception65_humanseg)

&emsp;&emsp;该示例展示了利用deeplabv3p_xception65_humanseg完成图像分割服务化部署和在线预测，获取识别结果和分割后的图像。

* [中文情感分析-基于simnet_bow](module_serving/semantic_model_simnet_bow)

&emsp;&emsp;该示例展示了利用senta_lstm完成中文文本情感分析服务化部署和在线预测，获取文本的情感分析结果。

关于Paddle-Serving一键模型部署功能的具体信息请参见[serving](module_serving)
