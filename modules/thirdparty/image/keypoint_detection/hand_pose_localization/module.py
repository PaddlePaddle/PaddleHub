# coding=utf-8
import os

from paddlehub import Module
from paddlehub.module.module import moduleinfo, serving

from hand_pose_localization.model import Model
from hand_pose_localization.processor import base64_to_cv2, Processor


@moduleinfo(
    name="hand_pose_localization",  # 模型名称
    type="CV/keypoint_detection",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="hand_pose_localization",  # 模型介绍
    version="1.0.2"  # 版本号
)
class Hand_Pose_Localization(Module):
    # 初始化函数
    def __init__(self, name=None, use_gpu=False):
        # 设置模型路径
        self.model_path = os.path.join(self.directory, "hand_pose_localization")

        # 加载模型
        self.model = Model(modelpath=self.model_path, use_gpu=use_gpu, use_mkldnn=False, combined=True)

    # 关键点检测函数
    def keypoint_detection(self, images=None, paths=None, batch_size=1, output_dir='output', visualization=False):
        # 加载数据处理器
        processor = Processor(images, paths, batch_size, output_dir)

        # 模型预测
        outputs = self.model.predict(processor.input_datas)

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
