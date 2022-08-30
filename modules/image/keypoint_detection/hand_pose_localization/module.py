# coding=utf-8
import os

import numpy as np
from paddlehub.module.module import moduleinfo, serving

from .model import InferenceModel
from .processor import base64_to_cv2, Processor


@moduleinfo(
    name="hand_pose_localization",  # 模型名称
    type="CV/keypoint_detection",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="hand_pose_localization",  # 模型介绍
    version="1.0.2"  # 版本号
)
class Hand_Pose_Localization:
    # 初始化函数
    def __init__(self, use_gpu=False):
        # 设置模型路径
        self.model_path = os.path.join(self.directory, "hand_pose_localization", "model")

        # 加载模型
        self.model = InferenceModel(modelpath=self.model_path, use_gpu=use_gpu)

        self.model.eval()

    # 关键点检测函数
    def keypoint_detection(self, images=None, paths=None, batch_size=1, output_dir='output', visualization=False):
        # 加载数据处理器
        processor = Processor(images, paths, batch_size, output_dir)

        # 模型预测
        outputs = []
        for input_data in processor.input_datas:
            output = self.model(input_data)
            outputs.append(output)
        outputs = np.concatenate(outputs, 0)

        # 结果后处理
        results = processor.postprocess(outputs, visualization)

        # 返回结果
        return results

    # Hub Serving
    @serving
    def serving_method(self, images, **kwargs):
        # 获取输入数据
        images_decode = [base64_to_cv2(image) for image in images]
        # 关键点检测
        results = self.keypoint_detection(images_decode, **kwargs)
        # 返回结果
        return results
