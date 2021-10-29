import os
import cv2
import paddle
import paddle.nn as nn
import numpy as np
from FCN_HRNet_W18_Face_Seg.model import FCN, HRNet_W18
from paddlehub.module.module import moduleinfo


@moduleinfo(
    name="FCN_HRNet_W18_Face_Seg",  # 模型名称
    type="CV",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="FCN_HRNet_W18_Face_Seg",  # 模型介绍
    version="1.0.0"  # 版本号
)
class FCN_HRNet_W18_Face_Seg(nn.Layer):
    def __init__(self):
        super(FCN_HRNet_W18_Face_Seg, self).__init__()
        # 加载分割模型
        self.seg = FCN(num_classes=2, backbone=HRNet_W18())

        # 加载模型参数
        state_dict = paddle.load(os.path.join(self.directory, 'seg_model_384.pdparams'))
        self.seg.set_state_dict(state_dict)

        # 设置模型为评估模式
        self.seg.eval()

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

    # 数据预处理函数
    @staticmethod
    def preprocess(images, batch_size):
        input_datas = []

        for image in images:
            # 图像缩放
            image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

            # 数据格式转换
            image = (image / 255.)[np.newaxis, :, :, :]
            image = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)

            input_datas.append(image)

        input_datas = np.concatenate(input_datas, 0)

        # 数据切分
        datas_num = input_datas.shape[0]
        split_num = datas_num // batch_size + 1 if datas_num % batch_size != 0 else datas_num // batch_size
        input_datas = np.array_split(input_datas, split_num)

        return input_datas

    # 结果归一化函数
    @staticmethod
    def normPRED(d):
        ma = np.max(d)
        mi = np.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    # 结果后处理函数
    def postprocess(self, outputs, datas, output_dir, visualization):
        # 检查输出目录
        if visualization:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        results = []

        for output, image, i in zip(outputs, datas, range(len(datas))):
            # 计算MASK
            pred = self.normPRED(output[1])

            # 图像缩放
            h, w = image.shape[:2]
            mask = cv2.resize(pred, (w, h))
            mask[mask < 0.5] = 0.
            mask[mask > 0.55] = 1.

            # 计算输出图像
            face = (image * mask[..., np.newaxis] + (1 - mask[..., np.newaxis]) * 255).astype(np.uint8)

            # 格式还原
            mask = (mask * 255).astype(np.uint8)

            # 可视化结果保存
            if visualization:
                cv2.imwrite(os.path.join(output_dir, 'result_mask_%d.png' % i), mask)
                cv2.imwrite(os.path.join(output_dir, 'result_%d.png' % i), face)

            results.append({'mask': mask, 'face': face})

        return results

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []

        for data in input_datas:
            # 转换数据为Tensor
            data = paddle.to_tensor(data)

            # 模型前向计算
            logits = self.seg(data)

            outputs.append(logits[0].numpy())

        outputs = np.concatenate(outputs, 0)

        return outputs

    def Segmentation(self, images=None, paths=None, batch_size=1, output_dir='output', visualization=False):
        # 获取输入数据
        datas = self.load_datas(paths, images)

        # 数据预处理
        input_datas = self.preprocess(datas, batch_size)

        # 模型预测
        outputs = self.predict(input_datas)

        # 结果后处理
        results = self.postprocess(outputs, datas, output_dir, visualization)

        return results
