# 模型概述

- 本模型为单目深度估计模型，可用于户外道路图像的深度估计，估计效果如下：

<img src="demo/demo.png" style="float:left;" />

- 本模型算法基于CVPR 2017的[SfMlearner](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhou_Unsupervised_Learning_of_CVPR_2017_paper.html)，模型参数来自于[SfMlearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch)

# 模型使用

- 安装SfMlearner模块：`hub install SfMlearner`

## 命令行预测

- `hub run SfMlearner [参数]`
  - `--paths`：待预测图像的地址，注意图像不要出现同名
  - `--output_dir`：预测结果的输出文件夹，默认是'./output/'
  - `--output_disp`：是否输出disparity map，默认是True，加入该参数后为False
  - `output_depth`：是否输出depth map，默认是False，加入该参数后为True
  - `--use_gpu`：是否使用GPU，默认是False，加入该参数后为True
- 示例：`hub run SfMlearner --paths './input/t1.jpg' --output_dir './output/' --output_depth --use_gpu`

## 脚本预测

- 预测代码示例：

```python
import paddlehub as hub
from PIL import Image
import numpy as np

raw = Image.open('./input/t1.jpg')
img = np.array(raw)  # 形状为[H, W, C]

SfMlearner = hub.Module(name='SfMlearner')
SfMlearner.estimation(images=img,
                      paths='./input/t1.png',
                      output_dir='./output/',
                      use_gpu=False,
                      output_disp = True,
                      output_depth=False)
```

- `SfMlearner.estimation`参数说明：
  - images(np.ndarray or list[np.ndarray])：图片数据，每个元素大小为[H, W, C]，默认为None
  - paths(str or list[str])：图片的路径，默认为None，允许同时输入images和paths
  - output_dir(str)：输出结果的文件夹，预测结果大小统一为128*416
  - use_gpu(bool)：是否采用GPU进行预测
  - output_disp(bool)：是否输出disparity map（即样例），默认为True
  - output_depth(bool)：是否输出depth map（效果不如disparity map），默认为False

# 库依赖

- paddlepaddle >= 2.1.0
- paddlehub >= 2.1.0
- PIL
- numpy
- os
- argparse
- matplotlib



