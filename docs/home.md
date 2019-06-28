欢迎使用**PaddleHub**！

PaddleHub是PaddlePaddle生态下的预训练模型的管理工具，旨在让PaddlePaddle生态下的开发者更便捷享受到大规模预训练模型的价值。通过PaddleHub，用户可以便捷地获取PaddlePaddle生态下的预训练模型，完成模型的管理和一键预测。此外，利用PaddleHub Fine-tune API，用户可以基于大规模预训练模型快速实现迁移学习，让预训练模型能更好服务于用户特定场景的应用。

![PaddleHub](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/paddlehub_figure.jpg)

PaddleHub主要包括两个功能：
## 命令行工具：

借鉴了Anaconda和PIP等软件包管理的理念，开发了PaddleHub命令行工具。可以方便快捷的完成模型的搜索、下载、安装、升级、预测等功能。
更加详细的使用说明可以参考
[PaddleHub命令行工具](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/cmd_tool.md)。

目前的预训练模型覆盖了图像分类、目标检测、词法分析、Transformer、情感分析五大类。
未来会持续开放更多类型的深度学习模型，如语言模型、视频分类、图像生成等供开发者更便捷的使用PaddlePaddle生态下的预训练模型。


## Fine-tune API:

通过PaddleHub Fine-tune API，开发者可以更便捷地让预训练模型能更好服务于特定场景的应用。大规模预训练模型结合Fine-tuning，可以在更短的时间完成模型的训练，同时模型具备更好的泛化能力。

![PaddleHub-Finetune](https://raw.githubusercontent.com/PaddlePaddle/PaddleHub/release/v0.5.0/docs/imgs/paddlehub_finetune.jpg)

更多关于Fine-tune API的详细信息和应用案例可以参考：

* [PaddleHub Fine-tune API](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/api/finetune_api.md)

* [PaddleHub文本分类迁移教程](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/nlp_tl_turtorial.md)

* [PaddleHub图像分类迁移教程](https://github.com/PaddlePaddle/PaddleHub/tree/release/v0.5.0/docs/turtorial/cv_tl_turtorial.md)
