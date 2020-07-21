## 命令行预测

```shell
$ hub run yolov3_darknet53_venus --input_path "/PATH/TO/IMAGE"
```

## API

```python
def context(trainable=True,
            pretrained=True,
            get_prediction=False)
```

提取特征，用于迁移学习。

**参数**

* trainable(bool): 参数是否可训练；
* pretrained (bool): 是否加载预训练模型；
* get\_prediction (bool): 是否执行预测。

**返回**

* inputs (dict): 模型的输入，keys 包括 'image', 'im\_size'，相应的取值为：
    * image (Variable): 图像变量
    * im\_size (Variable): 图片的尺寸
* outputs (dict): 模型的输出。如果 get\_prediction 为 False，输出 'head\_features'、'body\_features'，否则输出 'bbox\_out'。
* context\_prog (Program): 用于迁移学习的 Program.

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
