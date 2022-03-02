import os
import cv2
import numpy as np

from paddlehub import Module
from paddlehub.module.module import moduleinfo

from paddle.vision.transforms import Compose
from MiDaS_Small.utils import write_depth
from MiDaS_Small.inference import InferenceModel
from MiDaS_Small.transforms import Resize, NormalizeImage, PrepareForNet


@moduleinfo(
    name="MiDaS_Small",  # 模型名称
    type="CV/style_transfer",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="MiDaS_Small",  # 模型介绍
    version="1.0.0"  # 版本号
)
class MiDaS_Small(Module):
    # 初始化函数
    def __init__(self, name=None, directory=None, use_gpu=False):
        # 设置模型路径
        model_path = os.path.join(self.directory, "model-small")

        # 加载模型
        self.model = InferenceModel(modelpath=model_path, use_gpu=use_gpu, use_mkldnn=False, combined=True)
        self.model.eval()

        # 数据预处理配置
        self.net_h, self.net_w = 256, 256
        self.transform = Compose([
            Resize(
                self.net_w,
                self.net_h,
                resize_target=None,
                keep_aspect_ratio=False,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])

    # 数据读取函数
    @staticmethod
    def load_datas(paths, images):
        datas = []

        # 读取数据列表
        if paths is not None:
            for im_path in paths:
                assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
                im = cv2.imread(im_path)
                datas.append(im)

        if images is not None:
            datas = images

        # 返回数据列表
        return datas

    # 数据预处理函数
    def preprocess(self, datas):
        input_datas = []

        for img in datas:
            # 归一化
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

            # 图像变换
            img = self.transform({"image": img})["image"]

            # 新增维度
            input_data = img[np.newaxis, ...]

            input_datas.append(input_data)

        # 拼接数据
        input_datas = np.concatenate(input_datas, 0)

        return input_datas

    # 数据后处理函数
    @staticmethod
    def postprocess(datas, results, output_dir='output', visualization=False):
        # 检查输出目录
        if visualization:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        outputs = []

        for img, result, count in zip(datas, results, range(len(datas))):
            # 缩放回原尺寸
            output = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

            # 可视化输出
            if visualization:
                pfm_f, png_f = write_depth(os.path.join(output_dir, str(count)), output, bits=2)

            outputs.append(output)

        return outputs

    # 深度估计函数
    def depth_estimation(self, images=None, paths=None, batch_size=1, output_dir='output', visualization=False):
        # 加载数据
        datas = self.load_datas(paths, images)

        # 数据预处理
        input_datas = self.preprocess(datas)

        # 模型预测
        results = self.model(input_datas, batch_size=batch_size)[0]

        # 结果后处理
        outputs = self.postprocess(datas, results, output_dir, visualization)

        return outputs
