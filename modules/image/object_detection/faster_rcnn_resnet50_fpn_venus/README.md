## 命令行预测

```shell
$ hub run faster_rcnn_resnet50_fpn_venus --input_path "/PATH/TO/IMAGE"
```

## API

```python
def context(num_classes=81,
            trainable=True,
            pretrained=True,
            phase='train')
```

提取特征，用于迁移学习。

**参数**

* num\_classes (int): 类别数；
* trainable(bool): 参数是否可训练；
* pretrained (bool): 是否加载预训练模型；
* phase (str): 可选值为 'train'/'predict'，'trian' 用于训练，'predict' 用于预测。

**返回**

* inputs (dict): 模型的输入，相应的取值为：
    当 phase 为 'train'时，包含：
        * image (Variable): 图像变量
        * im\_size (Variable): 图像的尺寸
        * im\_info (Variable): 图像缩放信息
        * gt\_class (Variable): 检测框类别
        * gt\_box (Variable): 检测框坐标
        * is\_crowd (Variable): 单个框内是否包含多个物体
    当 phase 为 'predict'时，包含：
        * image (Variable): 图像变量
        * im\_size (Variable): 图像的尺寸
        * im\_info (Variable): 图像缩放信息
* outputs (dict): 模型的输出，相应的取值为：
    当 phase 为 'train'时，包含：
        * head_features (Variable): 所提取的特征
        * rpn\_cls\_loss (Variable): 检测框分类损失
        * rpn\_reg\_loss (Variable): 检测框回归损失
        * generate\_proposal\_labels (Variable): 图像信息
    当 phase 为 'predict'时，包含：
        * head_features (Variable): 所提取的特征
        * rois (Variable): 提取的roi
        * bbox\_out (Variable): 预测结果
* context\_prog (Program): 用于迁移学习的 Program。

```python
def save_inference_model(dirname,
                         model_filename=None,
                         params_filename=None,
                         combined=True)
```

将模型保存到指定路径。

**参数**

* dirname: 存在模型的目录名称
* model\_filename: 模型文件名称，默认为\_\_model\_\_
* params\_filename: 参数文件名称，默认为\_\_params\_\_(仅当`combined`为True时生效)
* combined: 是否将参数保存到统一的一个文件中

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
