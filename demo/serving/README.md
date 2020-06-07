# PaddleHub Serving
## 简介
### 背景
使用PaddleHub能够完成预训练模型的管理和预测，但开发者还经常面临将模型部署上线以对外提供服务的需求，而利用PaddleHub Serving可便捷的将模型部署上线，开发者只需要关注如何处理输入数据和输出结果即可。
### 主要功能
PaddleHub Serving是基于PaddleHub的一键模型服务部署工具，能够通过简单的Hub命令行工具轻松启动一个模型预测在线服务。

PaddleHub Serving主要包括利用Bert Service实现embedding服务化，以及利用预测模型实现预训练模型预测服务化两大功能，未来还将支持开发者使用PaddleHub Fine-tune API的模型服务化。

## Bert Service
`Bert Service`是基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)框架的快速部署模型远程计算服务方案，可将embedding过程通过调用API接口的方式实现，减少了对机器资源的依赖。使用PaddleHub可在服务器上一键部署`Bert Service`服务，在另外的普通机器上通过客户端接口即可轻松的获取文本对应的embedding数据。

关于其具体信息和demo请参见[Bert Service](../../docs/tutorial/bert_service.md)

该示例展示了利用`Bert Service`进行远程embedding服务化部署和在线预测，并获取文本embedding结果。

##  预训练模型一键服务部署
预训练模型一键服务部署是基于PaddleHub的预训练模型快速部署的服务化方案，能够将模型预测以API接口的方式实现。

关于预训练模型一键服务部署的具体信息请参见[PaddleHub Serving](../../docs/tutorial/serving.md)

预训练模型一键服务部署包括以下示例：

* [中文词法分析-基于lac](module_serving/lexical_analysis_lac)

&emsp;&emsp;该示例展示了利用lac完成中文文本分词服务化部署和在线预测，获取文本的分词结果，并可通过用户自定义词典干预分词结果。

* [人脸检测-基于pyramidbox_lite_server_mask](module_serving/object_detection_pyramidbox_lite_server_mask)

&emsp;&emsp;该示例展示了利用pyramidbox_lite_server_mask完成人脸口罩检测，检测人脸位置以及戴口枣的置信度。
