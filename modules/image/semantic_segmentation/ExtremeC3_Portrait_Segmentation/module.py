import os
import cv2
import paddle
import numpy as np

from paddle.nn import Layer
from paddlehub.module.module import moduleinfo

from ExtremeC3_Portrait_Segmentation.model import ExtremeC3Net

@moduleinfo(
    name="ExtremeC3_Portrait_Segmentation",  # 模型名称
    type="CV/semantic_segmentation",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="ExtremeC3_Portrait_Segmentation",  # 模型介绍
    version="1.0.0"  # 版本号
)
class ExtremeC3_Portrait_Segmentation(Layer):
    # 初始化函数
    def __init__(self, name=None, directory=None):
        super(ExtremeC3_Portrait_Segmentation, self).__init__()
        # 设置模型路径
        self.model_path = os.path.join(self.directory, "ExtremeC3.pdparams")

        # 加载模型
        self.model = ExtremeC3Net(classes=1, p=1, q=5)
        self.model.set_dict(paddle.load(self.model_path))
        self.model.eval()

        # 均值方差
        self.mean = [107.304565, 115.69884, 132.35703 ]
        self.std = [63.97182, 65.1337, 68.29726]

    # 读取数据函数
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

    # 预处理函数
    def preprocess(self, datas, batch_size):
        input_datas = []
        for img in datas:
            # 缩放
            h, w = img.shape[:2]
            img = cv2.resize(img, (224, 224))

            # 格式转换
            img = img.astype(np.float32)

            # 归一化
            for j in range(3):
                img[:, :, j] -= self.mean[j]
            for j in range(3):
                img[:, :, j] /= self.std[j]
            img /= 255.

            # 格式转换
            img = img.transpose((2, 0, 1))
            img = img[np.newaxis, ...]

            input_datas.append(img)

        # 数据切分
        input_datas = np.concatenate(input_datas, 0)
        split_num = len(datas) // batch_size + 1 if len(datas) % batch_size != 0 else len(datas) // batch_size
        input_datas = np.array_split(input_datas, split_num)
        return input_datas

    # 后处理函数
    def postprocess(self, outputs, datas, output_dir, visualization):
        # 检查输出目录
        if visualization:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        # 拼接输出
        outputs = paddle.concat(outputs)

        results = []
        for output, img, i in zip(outputs, datas, range(len(datas))):
            # 计算MASK
            mask = (output[0] > 0).numpy().astype('float32')

            # 缩放
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h))

            # 计算输出图像
            result = (img * mask[..., np.newaxis] + (1 - mask[..., np.newaxis]) * 255).astype(np.uint8)

            # 格式还原
            mask = (mask * 255).astype(np.uint8)

            # 可视化
            if visualization:
                cv2.imwrite(os.path.join(output_dir, 'result_mask_%d.png' % i), mask)
                cv2.imwrite(os.path.join(output_dir, 'result_%d.png' % i), result)

            results.append({
                'mask': mask,
                'result': result
            })

        return results

    # 关键点检测函数
    def Segmentation(self,
                       images=None,
                       paths=None,
                       batch_size=1,
                       output_dir='output',
                       visualization=False):
        # 加载数据处理器
        datas = self.load_datas(paths, images)

        # 获取输入数据
        input_datas = self.preprocess(datas, batch_size)

        # 模型预测
        outputs = [self.model(paddle.to_tensor(input_data)) for input_data in input_datas]

        # 结果后处理
        results = self.postprocess(outputs, datas, output_dir, visualization)

        # 返回结果
        return results
