# PaddleHub v0.5.0

正式发布PaddleHub预训练模型管理工具，旨在帮助用户更高效的管理模型并开展迁移学习的工作。

* 预训练模型管理: 通过hub命令行可完成PaddlePaddle生态的预训练模型下载、搜索、版本管理等功能。

* 命令行一键使用: 无需代码，通过命令行即可直接使用预训练模型进行预测，快速调研训练模型效果。
                目前版本支持以下模型：词法分析LAC；情感分析Senta；目标检测SSD；图像分类ResNet, MobileNet, NASNet等。

* 迁移学习: 提供了基于预训练模型的Finetune API，用户通过少量代码即可完成迁移学习，包括BERT/ERNIE文本分类、序列标注、图像分类迁移等。

# PaddleHub v1.0.0

* 全新发布[**PaddleHub官网**](https://www.paddlepaddle.org.cn/hub)，易用性全面提升
* **新增29个预训练模型**，覆盖文本、图像、视频三大领域；目前官方提供40个预训练模型
* Fine-tune API升级，灵活性与性能全面提升

# PaddleHub v1.0.1

* 安装模型时自动选择与paddlepaddle版本适配的模型

# PaddleHub v1.1.0

* PaddleHub **新增预训练模型ERNIE 2.0**

# PaddleHub v1.1.1

* PaddleHub支持离线运行
* 修复python2安装PaddleHub失败问题

# PaddleHub v1.2.0

* 新增**超参优化Auto Fine-tune**，详情参考[使用说明](https://github.com/PaddlePaddle/PaddleHub/blob/develop/tutorial/autofinetune.md)
* 新增**阅读理解任务**和**回归任务**
* 新增Fine-tune**优化策略ULMFiT**
