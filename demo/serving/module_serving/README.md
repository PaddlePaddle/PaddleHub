# PaddleHub Serving模型一键服务部署
## 简介
### 为什么使用一键服务部署
使用PaddleHub能够快速进行模型预测，但开发者常面临本地预测过程迁移线上的需求。无论是对外开放服务端口，还是在局域网中搭建预测服务，都需要PaddleHub具有快速部署模型预测服务的能力。在这个背景下，模型一键服务部署工具——PaddleHub Serving应运而生。开发者通过一行命令即可快速启动一个模型预测在线服务，而无需关注网络框架选择和实现。
### 什么是一键服务部署
PaddleHub Serving是基于PaddleHub的一键模型服务部署工具，能够通过简单的Hub命令行工具轻松启动一个模型预测在线服务，前端通过Flask和Gunicorn完成网络请求的处理，后端直接调用PaddleHub预测接口，同时支持使用多进程方式利用多核提高并发能力，保证预测服务的性能。

### 支持模型
目前PaddleHub Serving支持PaddleHub所有可直接用于预测的模型进行服务部署，包括`lac`、`senta_bilstm`等NLP类模型，以及`yolov3_darknett53_coco2017`、`vgg16_imagenet`等CV类模型，未来还将支持开发者使用PaddleHub Fine-tune API得到的模型用于快捷服务部署。

**NOTE:** 关于PaddleHub Serving一键服务部署的具体信息请参见[PaddleHub Serving](../../../docs/tutorial/serving.md)。

## Demo

获取PaddleHub Serving的一键服务部署场景示例，可参见下列demo：

* [中文词法分析-基于lac](../module_serving/lexical_analysis_lac)

&emsp;&emsp;该示例展示了利用lac完成中文文本分词服务化部署和在线预测，获取文本的分词结果，并可通过用户自定义词典干预分词结果。

* [人脸检测-基于pyramidbox_lite_server_mask](../module_serving/object_detection_pyramidbox_lite_server_mask)

&emsp;&emsp;该示例展示了利用pyramidbox_lite_server_mask完成人脸口罩检测，检测人脸位置以及戴口枣的置信度。

## Bert Service
除了预训练模型一键服务部署功能之外，PaddleHub Serving还具有`Bert Service`功能，支持ernie_tiny、bert等模型快速部署，对外提供可靠的在线embedding服务，具体信息请参见[Bert Service](../../../docs/tutorial/bert_service.md)。
