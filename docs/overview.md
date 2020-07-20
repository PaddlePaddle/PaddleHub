欢迎使用**PaddleHub**！

# 概述

## PaddleHub是什么

PaddleHub是飞桨生态的预训练模型应用工具，开发者可以便捷地使用高质量的预训练模型结合Fine-tune API快速完成模型迁移到部署的全流程工作。PaddleHub提供的预训练模型涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型。更多详情可查看[PaddleHub官网](https://www.paddlepaddle.org.cn/hub)

PaddleHub以预训练模型应用为核心具备以下特点：  

* **[模型即软件](#模型即软件)**，通过Python API或命令行实现模型调用，可快速体验或集成飞桨特色预训练模型。

* **[易用的迁移学习](#易用的迁移学习)**，通过Fine-tune API，内置多种优化策略，只需少量代码即可完成预训练模型的Fine-tuning。

* **[一键模型转服务](#一键模型转服务)**，简单一行命令即可搭建属于自己的深度学习模型API服务完成部署。

* **[自动超参优化](#自动超参优化)**，内置AutoDL Finetuner能力，一键启动自动化超参搜索。

<p align="center">
 <img src="imgs/paddlehub_figure.jpg" width='70%' align="middle"  
</p>



## PaddleHub特性

### 一、模型即软件

PaddleHub采用模型即软件的设计理念，所有的预训练模型与Python软件包类似，具备版本的概念，通过`hub install/uninstall` 可以便捷完成模型的升级和卸载。此外，还可以通过Python的API或命令行实现快速预测的软件集成，更方便地应用和管理深度学习模型，完成模型的搜索、下载、安装、升级、预测等功能。更加详细的使用说明可以参考[PaddleHub命令行工具](tutorial/cmdintro.md)。

目前的预训练模型覆盖了图像分类、语义模型、视频分类、图像生成、图像分割、文本审核、关键点检测等业界主流模型，更多PaddleHub已经发布的模型，请前往 [PaddleHub官网](https://www.paddlepaddle.org.cn/hub) 查看。也可以使用[快速体验](quickstart.md)通过命令行即可调用预训练模型进行预测。

### 二、易用的迁移学习

迁移学习(Transfer Learning)通俗来讲，就是运用已有的知识来学习新的知识，核心是找到已有知识和新知识之间的相似性。PaddleHub提供了Fine-tune API，只需要少量代码即可完成深度学习模型在自然语言处理和计算机视觉场景下的迁移学习，可以在更短的时间完成模型的训练，同时模型具备更好的泛化能力。

<p align="center">
 <img src="imgs/paddlehub_finetune.jpg" width='70%' align="middle"  
</p>

<p align="center">
 <img src="imgs/paddlehub_finetune.gif" align="middle"  
</p>

<p align='center'>
 十行代码完成ERNIE工业级文本分类
</p>


  PaddleHub提供了使用Finetune-API和预训练模型完成[文本分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/text_classification)、[序列标注](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/sequence_labeling)、[多标签分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/multi_label_classification)、[图像分类](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/image_classification)、[检索式问答任务](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/qa_classification)、[回归任务](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/regression)、[句子语义相似度计算](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/sentence_similarity)、[阅读理解任务](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo/reading_comprehension)等迁移任务的使用示例，详细参见[demo](https://github.com/PaddlePaddle/PaddleHub/tree/release/v1.7/demo)。

* 场景化使用

  PaddleHub在AI Studio上提供了IPython NoteBook形式的demo。用户可以直接在平台上在线体验，链接如下：

|预训练模型|任务类型|数据集|AIStudio链接|备注|
|-|-|-|-|-|
|ResNet|图像分类|猫狗数据集DogCat|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147010)||
|ERNIE|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147006)||
|ERNIE|文本分类|中文新闻分类数据集THUNEWS|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/221999)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成文本分类迁移学习。|
|ERNIE|序列标注|中文序列标注数据集MSRA_NER|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/147009)||
|ERNIE|序列标注|中文快递单数据集Express|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/184200)|本教程讲述了如何将自定义数据集加载，并利用Fine-tune API完成序列标注迁移学习。|
|ERNIE Tiny|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/186443)||
|Senta|文本分类|中文情感分类数据集ChnSentiCorp|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/216846)|本教程讲述了任何利用Senta和Fine-tune API完成情感分类迁移学习。|
|Senta|情感分析预测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215814)||
|LAC|词法分析|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215711)||
|Ultra-Light-Fast-Generic-Face-Detector-1MB|人脸检测|N/A|[点击体验](https://aistudio.baidu.com/aistudio/projectdetail/215962)||

**NOTE:** [`飞桨PaddleHub`](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/79927)是PaddleHub的官方账号。

关于PaddleHub快捷完成迁移学习，更多信息参考：

[API](reference)

[自定义数据集如何Fine-tune](tutorial/how_to_load_data.md)

[实现自定义迁移任务](tutorial/how_to_define_task.md)

[ULMFiT优化策略](tutorial/strategy_exp.md)

### 三、一键模型转服务

PaddleHub Serving是基于PaddleHub的一键模型服务部署工具，能够通过简单的Hub命令行工具轻松启动一个模型预测在线服务。
其主要包括利用Bert Service实现embedding服务化，以及利用预测模型实现预训练模型预测服务化两大功能。未来还将支持开发者使用PaddleHub Fine-tune API的模型服务化。

关于服务化部署详细信息参见[PaddleHub Serving一键服务部署](tutorial/serving.md)。

### 四、自动超参优化

深度学习模型往往包含许多的超参数，而这些超参数的取值对模型性能起着至关重要的作用。因为模型参数空间大，目前超参调整都是通过手动，依赖人工经验或者不断尝试，且不同模型、样本数据和场景下不尽相同，所以需要大量尝试，时间成本和资源成本非常浪费。PaddleHub AutoDL Finetuner可以实现自动调整超参数，使得模型性能达到最优水平。它通过多种调优的算法来搜索最优超参。

AutoDL Finetuner详细信息参见[PaddleHub超参优化](tutorial/autofinetune.md)。
