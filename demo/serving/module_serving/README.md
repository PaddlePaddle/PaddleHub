# PaddleHub Serving模型一键服务部署
## 简介
### 为什么使用一键服务部署
使用PaddleHub能够快速进行模型预测，但开发者常面临本地预测过程迁移线上的需求。无论是对外开放服务端口，还是在局域网中搭建预测服务，都需要PaddleHub具有快速部署模型预测服务的能力。在这个背景下，模型一键服务部署工具——PaddleHub Serving应运而生。开发者通过一句命令即可快速启动一个模型预测在线服务，而无需关注网络框架选择和实现。
### 什么是一键服务部署
PaddleHub Serving是基于PaddleHub的一键模型服务部署工具，能够通过简单的Hub命令行工具轻松启动一个模型预测在线服务，前端通过Flask和Gunicorn完成网络请求的处理，后端直接调用PaddleHub预测接口，同时支持使用多进程方式利用多核提高并发能力，保证预测服务的性能。

### 支持模型
目前PaddleHub Serving支持PaddleHub所有可直接用于预测的模型进行服务部署，包括`lac`、`senta_bilstm`等NLP类模型，以及`yolov3_darknett53_coco2017`、`vgg16_imagenet`等CV类模型，未来还将支持开发者使用PaddleHub Fine-tune API得到的模型用于快捷服务部署。

**NOTE:** 关于PaddleHub Serving一键服务部署的具体信息请参见[PaddleHub Serving](../../../tutorial/serving.md)。

## Demo

获取PaddleHub Serving的一键服务部署场景示例，可参见下列demo：

* [图像分类-基于vgg11_imagent](../module_serving/classification_vgg11_imagenet)  

&emsp;&emsp;该示例展示了利用vgg11_imagent完成图像分类服务化部署和在线预测，获取图像分类结果。

* [图像生成-基于stgan_celeba](../module_serving/GAN_stgan_celeba)  

&emsp;&emsp;该示例展示了利用stgan_celeba生成图像服务化部署和在线预测，获取指定风格的生成图像。

* [文本审核-基于porn_detection_lstm](../module_serving/text_censorship_porn_detection_lstm)  

&emsp;&emsp;该示例展示了利用porn_detection_lstm完成中文文本黄色敏感信息鉴定的服务化部署和在线预测，获取文本是否敏感及其置信度。

* [中文词法分析-基于lac](../module_serving/lexical_analysis_lac)

&emsp;&emsp;该示例展示了利用lac完成中文文本分词服务化部署和在线预测，获取文本的分词结果，并可通过用户自定义词典干预分词结果。

* [目标检测-基于yolov3_darknet53_coco2017](../module_serving/object_detection_yolov3_darknet53_coco2017)  

&emsp;&emsp;该示例展示了利用yolov3_darknet53_coco2017完成目标检测服务化部署和在线预测，获取检测结果和覆盖识别框的图片。

* [中文语义分析-基于simnet_bow](../module_serving/semantic_model_simnet_bow)

&emsp;&emsp;该示例展示了利用simnet_bow完成中文文本相似度检测服务化部署和在线预测，获取文本的相似程度。  

* [图像分割-基于deeplabv3p_xception65_humanseg](../module_serving/semantic_segmentation_deeplabv3p_xception65_humanseg)

&emsp;&emsp;该示例展示了利用deeplabv3p_xception65_humanseg完成图像分割服务化部署和在线预测，获取识别结果和分割后的图像。

* [中文情感分析-基于simnet_bow](../module_serving/semantic_model_simnet_bow)

&emsp;&emsp;该示例展示了利用senta_lstm完成中文文本情感分析服务化部署和在线预测，获取文本的情感分析结果。

## Bert Service
除了预训练模型一键服务部署功能之外，PaddleHub Serving还具有`Bert Service`功能，支持ernie_tiny、bert等模型快速部署，对外提供可靠的在线embedding服务，具体信息请参见[Bert Service](../../../tutorial/bert_service.md)。
