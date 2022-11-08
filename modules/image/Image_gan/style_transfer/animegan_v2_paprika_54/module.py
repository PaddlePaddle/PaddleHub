import os

from .model import InferenceModel
from .processor import base64_to_cv2
from .processor import cv2_to_base64
from .processor import Processor
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import serving


@moduleinfo(
    name="animegan_v2_paprika_54",  # 模型名称
    type="CV/style_transfer",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="animegan_v2_paprika_54",  # 模型介绍
    version="1.1.0"  # 版本号
)
class Animegan_V2_Paprika_54:
    # 初始化函数
    def __init__(self, use_gpu=False, use_mkldnn=False):
        # 设置模型路径
        self.model_path = os.path.join(self.directory, "animegan_v2_paprika_54", "model")

        # 加载模型
        self.model = InferenceModel(modelpath=self.model_path, use_gpu=use_gpu, use_mkldnn=use_mkldnn)

        self.model.eval()

    # 关键点检测函数
    def style_transfer(self,
                       images=None,
                       paths=None,
                       output_dir='output',
                       visualization=False,
                       min_size=32,
                       max_size=1024):
        # 加载数据处理器
        processor = Processor(images=images,
                              paths=paths,
                              batch_size=1,
                              output_dir=output_dir,
                              min_size=min_size,
                              max_size=max_size)

        # 模型预测
        outputs = []
        for input_data in processor.input_datas:
            output = self.model(input_data)
            outputs.append(output)

        # 结果后处理
        results = processor.postprocess(outputs, visualization)

        # 返回结果
        return results

    # Hub Serving
    @serving
    def serving_method(self, images, **kwargs):
        # 获取输入数据
        images_decode = [base64_to_cv2(image) for image in images]

        # 图片风格转换
        results = self.style_transfer(images_decode, **kwargs)

        # 对输出图片进行编码
        encodes = []
        for result in results:
            encode = cv2_to_base64(result)
            encodes.append(encode)

        # 返回结果
        return encodes
