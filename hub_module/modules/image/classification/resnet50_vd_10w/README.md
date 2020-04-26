<p align="center">
<img src="http://bj.bcebos.com/ibox-thumbnail98/77fa9b7003e4665867855b2b65216519?authorization=bce-auth-v1%2Ffbe74140929444858491fbf2b6bc0935%2F2020-04-08T11%3A05%3A10Z%2F1800%2F%2F1df0ecb4a52adefeae240c9e2189e8032560333e399b3187ef1a76e4ffa5f19f"  hspace='5' width=800/> <br /> ResNet 系列的网络结构
</p>

模型的详情可参考[论文](https://arxiv.org/pdf/1812.01187.pdf)


## API

```python
def get_expected_image_width()
```

返回预处理的图片宽度，也就是224。

```python
def get_expected_image_height()
```

返回预处理的图片高度，也就是224。

```python
def get_pretrained_images_mean()
```

返回预处理的图片均值，也就是 \[0.485, 0.456, 0.406\]。

```python
def get_pretrained_images_std()
```

返回预处理的图片标准差，也就是 \[0.229, 0.224, 0.225\]。


```python
def context(trainable=True, pretrained=True)
```

**参数**

* trainable (bool): 计算图的参数是否为可训练的；
* pretrained (bool): 是否加载默认的预训练模型。

**返回**

* inputs (dict): 计算图的输入，key 为 'image', value 为图片的张量；
* outputs (dict): 计算图的输出，key 为 'feature\_map', value为全连接层前面的那个张量。
* context\_prog(fluid.Program): 计算图，用于迁移学习。


```python
def save_inference_model(dirname,
                         model_filename=None,
                         params_filename=None,
                         combined=True)
```

将模型保存到指定路径。

**参数**

* dirname: 存在模型的目录名称
* model_filename: 模型文件名称，默认为\_\_model\_\_
* params_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
* combined: 是否将参数保存到统一的一个文件中

## 代码示例

```python
import paddlehub as hub
import cv2

classifier = hub.Module(name="resnet50_vd_10w")
input_dict, output_dict, program = classifier.context(trainable=True)
```

### 查看代码

[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
