# SSD 目标检测

本示例展示如何使用[SSD](https://www.paddlepaddle.org.cn/hubdetail?name=ssd_mobilenet_v1_pascal&en_category=ObjectDetection)预训练模型进行目标检测预测。

SSD是一个目标检测模型，可以检测出图片中的实物的类别和位置，PaddleHub发布的SSD模型通过pascalvoc数据集训练，支持20个数据类别的检测。

## 命令行方式预测

```shell
$ hub run ssd_mobilenet_v1_pascal --input_path "/PATH/TO/IMAGE"
$ hub run ssd_mobilenet_v1_pascal --input_file test.txt
```

test.txt 存放待检测图片的存放路径。

## 通过python API预测

`ssd_demo.py`给出了使用python API调用SSD预测的示例代码。
通过以下命令试验下效果：

```shell
python ssd_demo.py
```
