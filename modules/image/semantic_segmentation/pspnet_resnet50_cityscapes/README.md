# pspnet_resnet50_cityscapes

|模型名称|pspnet_resnet50_cityscapes|
| :--- | :---: | 
|类别|图像-图像分割|
|网络|pspnet_resnet50vd|
|数据集|Cityscapes|
|是否支持Fine-tuning|是|
|模型大小|390MB|
|指标|-|
|最新更新日期|2022-03-21|

## 一、模型基本信息
  
  - 样例结果示例：
    <p align="center">
    <img src="https://user-images.githubusercontent.com/35907364/159212111-df341f2a-e994-45d7-92d6-2288d666079c.png"  width = "420" height = "505" hspace='10'/> <img src="https://user-images.githubusercontent.com/35907364/159212188-2db40b29-2943-47ce-9ad2-36a6fb85ba3e.png" width = "420" height = "505" hspace='10'/>
    </p>

- ### 模型介绍

    - 本示例将展示如何使用PaddleHub对预训练模型进行finetune并完成预测任务。
    - 更多详情请参考：[pspnet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf)

## 二、安装

- ### 1、环境依赖

    - paddlepaddle >= 2.0.0

    - paddlehub >= 2.0.0

- ### 2、安装

    - ```shell
      $ hub install pspnet_resnet50_cityscapes
      ```

    -  如您安装时遇到问题，可参考：[零基础windows安装](../../../../docs/docs_ch/get_start/windows_quickstart.md)
    | [零基础Linux安装](../../../../docs/docs_ch/get_start/linux_quickstart.md) | [零基础MacOS安装](../../../../docs/docs_ch/get_start/mac_quickstart.md)   


## 三、模型API预测

- ### 1.预测代码示例

    - ```python
      import cv2
      import paddle
      import paddlehub as hub

      if __name__ == '__main__':
          model = hub.Module(name='pspnet_resnet50_cityscapes')
          img = cv2.imread("/PATH/TO/IMAGE")
          result = model.predict(images=[img], visualization=True)
      ```

- ### 2.如何开始Fine-tune

    - 在完成安装PaddlePaddle与PaddleHub后，通过执行`python train.py`即可开始使用pspnet_resnet50_cityscapes模型对OpticDiscSeg数据集进行Fine-tune。 `train.py`内容如下：

    - 代码步骤

        - Step1: 定义数据预处理方式
            - ```python
              from paddlehub.vision.segmentation_transforms import Compose, Resize, Normalize

              transform = Compose([Resize(target_size=(512, 512)), Normalize()])
              ```

            - `segmentation_transforms` 数据增强模块定义了丰富的针对图像分割数据的预处理方式，用户可按照需求替换自己需要的数据预处理方式。

        - Step2: 下载数据集并使用
            - ```python
              from paddlehub.datasets import OpticDiscSeg

              train_reader = OpticDiscSeg(transform, mode='train')
              ```
                - `transforms`: 数据预处理方式。
                - `mode`: `mode`: 选择数据模式，可选项有 `train`, `test`, `val`, 默认为`train`。

                - 数据集的准备代码可以参考 [opticdiscseg.py](../../paddlehub/datasets/opticdiscseg.py)。`hub.datasets.OpticDiscSeg()`会自动从网络下载数据集并解压到用户目录下`$HOME/.paddlehub/dataset`目录。

        - Step3: 加载预训练模型

            - ```python
              import paddlehub as hub

              model = hub.Module(name='pspnet_resnet50_cityscapes', num_classes=2, pretrained=None)
              ```
                - `name`: 选择预训练模型的名字。
                - `load_checkpoint`: 是否加载自己训练的模型，若为None，则加载提供的模型默认参数。

        - Step4: 选择优化策略和运行配置

            - ```python
              import paddle
              from paddlehub.finetune.trainer import Trainer

              scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=1000, power=0.9,  end_lr=0.0001)
              optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
              trainer = Trainer(model, optimizer, checkpoint_dir='test_ckpt_img_seg', use_gpu=True)
              trainer.train(train_reader, epochs=10, batch_size=4, log_interval=10, save_interval=4)
              ```
             
    - 模型预测

        - 当完成Fine-tune后，Fine-tune过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为Fine-tune时所选择的保存checkpoint的目录。我们使用该模型来进行预测。predict.py脚本如下：

            ```python
            import paddle
            import cv2
            import paddlehub as hub

            if __name__ == '__main__':
                model = hub.Module(name='pspnet_resnet50_cityscapes', pretrained='/PATH/TO/CHECKPOINT')
                img = cv2.imread("/PATH/TO/IMAGE")
                model.predict(images=[img], visualization=True)
            ```

            - 参数配置正确后，请执行脚本`python predict.py`。

            - **Args**
                * `images`:原始图像路径或BGR格式图片；
                * `visualization`: 是否可视化，默认为True；
                * `save_path`: 保存结果的路径，默认保存路径为'seg_result'。

                **NOTE:** 进行预测时，所选择的module，checkpoint_dir，dataset必须和Fine-tune所用的一样。

## 四、服务部署

- PaddleHub Serving可以部署一个在线图像分割服务。

- ### 第一步：启动PaddleHub Serving

    - 运行启动命令：

    - ```shell
      $ hub serving start -m pspnet_resnet50_cityscapes
      ```

    - 这样就完成了一个图像分割服务化API的部署，默认端口号为8866。

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
        url = "http://127.0.0.1:8866/predict/ginet_resnet50vd_voc"
        r = requests.post(url=url, headers=headers, data=json.dumps(data))
        mask = base64_to_cv2(r.json()["results"][0])
        ```

## 五、更新历史

* 1.0.0

  初始发布
