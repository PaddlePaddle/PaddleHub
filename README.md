# PaddleHub
PaddleHub旨在为PaddlePaddle提供一个简明易用的预训练模型管理框架。
使用PaddleHub，你可以：

1. 通过统一的方式，快速便捷的获取PaddlePaddle发布的预训练模型
2. 利用PaddleHub提供的接口，对预训练模型进行Transfer learning
3. 以命令行或者python代码调用的方式，使用预训练模型进行预测

除此之外，我们还提供了预训练模型的本地管理机制（类似于pip），用户可以通过命令行来管理本地的预训练模型
![图片](https://github.com/PaddlePaddle/PaddleHub/tree/develop/docs/cmd_demo.JPG)

想了解PaddleHub已经发布的模型，请查看[模型列表](https://github.com/PaddlePaddle/PaddleHub/tree/develop/docs/released_module_list.md)
# 安装
paddle hub直接通过pip进行安装（python3以上），使用如下命令来安装paddle hub
```
pip install paddle_hub
```
# 快速体验
通过下面的命令，来体验下paddle hub的魅力
```
#使用lac进行分词
hub run lac --input_text "今天是个好日子"
#使用senta进行情感分析
hub run senta --input_text "今天是个好日子"
```
# 深入了解Paddle Hub
* [命令行功能](https://github.com/PaddlePaddle/PaddleHub/tree/develop/docs/command_line_introduction.md)
* [Transfer Learning](https://github.com/PaddlePaddle/PaddleHub/tree/develop/docs/transfer_learning_turtorial.md)
* API
