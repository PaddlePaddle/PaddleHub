# user_guided_colorization

|模型名称|user_guided_colorization|
| :--- | :---: | 
|类别|图像-图像编辑|
|网络| Local and Global Hints Network |
|数据集|ILSVRC 2012|
|是否支持Fine-tuning|是|
|模型大小|131MB|
|指标|-|
|最新更新日期|2021-02-26|


## 一、模型基本信息

- ### 模型介绍

- ### 应用效果展示
  
  - 样例结果示例(左为原图，右为效果图)：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/136653401-6644bd46-d280-4c15-8d48-680b7eb152cb.png" width = "300" height = "450" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/136648959-40493c9c-08ec-46cd-a2a2-5e2038dcbfa7.png" width = "300" height = "450" hspace='10'/>
    </p>

  - user_guided_colorization 是基于"Real-Time User-Guided Image Colorization with Learned Deep Priors"的着色模型，该模型利用预先提供的着色块对图像进行着色。


## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装
    - ```shell
      $ hub install user_guided_colorization
      ```

    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
    | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)   

## 三、模型API预测

- ### 1.命令行预测

    ```shell
    $ hub run user_guided_colorization --input_path "/PATH/TO/IMAGE"
    ```
- ### 2.预测代码示例

    ```python
    import paddle
    import paddlehub as hub

    if __name__ == '__main__':

        model = hub.Module(name='user_guided_colorization')
        model.set_config(prob=0.1)
        result = model.predict(images=['/PATH/TO/IMAGE'])
    ```
- ### 3.如何开始Fine-tune

    - 在完成安装PaddlePaddle与PaddleHub后，通过执行`python train.py`即可开始使用user_guided_colorization模型对[Canvas](../../docs/reference/datasets.md#class-hubdatasetsCanvas)等数据集进行Fine-tune。

    - 代码步骤

        - Step1: 定义数据预处理方式
            - ```python
              import paddlehub.vision.transforms as T

              transform = T.Compose([T.Resize((256, 256), interpolation='NEAREST'),
                       T.RandomPaddingCrop(crop_size=176),
                       T.RGB2LAB()], to_rgb=True)
              ```

                - `transforms` 数据增强模块定义了丰富的数据预处理方式，用户可按照需求替换自己需要的数据预处理方式。

        - Step2: 下载数据集并使用
            - ```python
              from paddlehub.datasets import Canvas

              color_set = Canvas(transform=transform, mode='train')
              ```

                * `transforms`: 数据预处理方式。
                * `mode`: 选择数据模式，可选项有 `train`, `test`, `val`， 默认为`train`。

                * `hub.datasets.Canvas()` 会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。


        - Step3: 加载预训练模型

            - ```python
              model = hub.Module(name='user_guided_colorization', load_checkpoint=None)
              model.set_config(classification=True, prob=1)
              ```
                * `name`:加载模型的名字。
                * `load_checkpoint`: 是否加载自己训练的模型，若为None，则加载提供的模型默认参数。
                * `classification`: 着色模型分两部分训练，开始阶段`classification`设置为True, 用于浅层网络训练。训练后期将`classification`设置为False, 用于训练网络的输出层。
                * `prob`: 每张输入图不加一个先验彩色块的概率，默认为1，即不加入先验彩色块。例如，当`prob`设定为0.9时，一张图上有两个先验彩色块的概率为(1-0.9)*(1-0.9)*0.9=0.009.

        - Step4: 选择优化策略和运行配置

            ```python
            optimizer = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
            trainer = Trainer(model, optimizer, checkpoint_dir='img_colorization_ckpt_cls_1')
            trainer.train(color_set, epochs=201, batch_size=25, eval_dataset=color_set, log_interval=10, save_interval=10)
            ```


            - 运行配置

            - `Trainer` 主要控制Fine-tune的训练，包含以下可控制的参数:

                * `model`: 被优化模型；
                * `optimizer`: 优化器选择；
                * `use_vdl`: 是否使用vdl可视化训练过程；
                * `checkpoint_dir`: 保存模型参数的地址；
                * `compare_metrics`: 保存最优模型的衡量指标；

            - `trainer.train` 主要控制具体的训练过程，包含以下可控制的参数：

                * `train_dataset`: 训练时所用的数据集；
                * `epochs`: 训练轮数；
                * `batch_size`: 训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
                * `num_workers`: works的数量，默认为0；
                * `eval_dataset`: 验证集；
                * `log_interval`: 打印日志的间隔， 单位为执行批训练的次数。
                * `save_interval`: 保存模型的间隔频次，单位为执行训练的轮数。

    - 模型预测

        -   当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。 我们使用该模型来进行预测。predict.py脚本如下：

            - ```python
              import paddle
              import paddlehub as hub

              if __name__ == '__main__':
                  model = hub.Module(name='user_guided_colorization', load_checkpoint='/PATH/TO/CHECKPOINT')
                  model.set_config(prob=0.1)
                  result = model.predict(images=['house.png'])
              ```


            - **NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。若想获取油画风着色效果，请下载参数文件[油画着色](https://paddlehub.bj.bcebos.com/dygraph/models/canvas_rc.pdparams)

## 四、服务部署

- PaddleHub Serving可以部署一个在线着色任务服务。

- ### 第一步：启动PaddleHub Serving

    - 运行启动命令：

    - ```shell
      $ hub serving start -m user_guided_colorization
      ```

    - 这样就完成了一个着色任务服务化API的部署，默认端口号为8866。

    - **NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

- ### 第二步：发送预测请求

    - 配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

       ```python
       import requests
       import json
       import cv2
       import base64
       import numpy as np

       def cv2_to_base64(image):
           data = cv2.imencode('.jpg', image)[1]
           return base64.b64encode(data.tostring()).decode('utf8')

       def base64_to_cv2(b64str):
           data = base64.b64decode(b64str.encode('utf8'))
           data = np.fromstring(data, np.uint8)
           data = cv2.imdecode(data, cv2.IMREAD_COLOR)
           return data

       # 发送HTTP请求
       org_im = cv2.imread('/PATH/TO/IMAGE')
       data = {'images':[cv2_to_base64(org_im)]}
       headers = {"Content-type": "application/json"}
       url = "http://127.0.0.1:8866/predict/user_guided_colorization"
       r = requests.post(url=url, headers=headers, data=json.dumps(data))
       data = base64_to_cv2(r.json()["results"]['data'][0]['fake_reg'])
       cv2.imwrite('color.png', data)
       ```


## 五、更新历史

* 1.0.0

  初始发布


