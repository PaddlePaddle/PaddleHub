import os
import cv2
import time
import base64
import numpy as np

__all__ = ['base64_to_cv2', 'cv2_to_base64', 'Processor']


def check_dir(dir_path):
    # 目录检查函数
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    elif os.path.isfile(dir_path):
        os.remove(dir_path)
        os.makedirs(dir_path)


def base64_to_cv2(b64str):
    # base64转cv2函数
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.frombuffer(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


def cv2_to_base64(image):
    # cv2转base64函数
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


class Processor():
    # 初始化函数
    def __init__(self, images=None, paths=None, output_dir='output', batch_size=1):
        # 变量设置
        self.images = images
        self.paths = paths
        self.output_dir = output_dir
        self.batch_size = batch_size

        # 获取原始输入数据
        self.datas = self.load_datas()

        # 对原始输入数据进行预处理
        self.input_datas = self.preprocess()

    # 读取数据函数
    def load_datas(self):
        datas = []

        # 读取数据列表
        if self.paths is not None:
            for im_path in self.paths:
                assert os.path.isfile(im_path), "The {} isn't a valid file path.".format(im_path)
                im = cv2.imread(im_path)
                datas.append(im)

        if self.images is not None:
            datas = self.images

        # 返回数据列表
        return datas

    # 数据预处理函数
    def preprocess(self):
        input_datas = []

        # 数据预处理
        for i, img in enumerate(self.datas):
            # 图像缩放
            img = cv2.resize(img, (256, 256))

            # 归一化
            img = (img.astype('float32') / 255.0 - 0.5) / 0.5

            # 转置
            img = img.transpose((2, 0, 1))

            # 增加维度
            img = np.expand_dims(img, axis=0)

            # 加入输入数据列表
            input_datas.append(img)

        # 数据按batch_size切分
        input_datas = np.concatenate(input_datas, 0)
        split_num = len(self.datas) // self.batch_size + 1 if len(self.datas) % self.batch_size != 0 else len(
            self.datas) // self.batch_size
        input_datas = np.array_split(input_datas, split_num)

        # 返回预处理完成的数据
        return input_datas

    def postprocess(self, outputs, visualization):
        results = []

        for im_id, output in enumerate(outputs):
            # 图像后处理
            img = (output * 0.5 + 0.5) * 255.

            # 限幅
            img = np.clip(img, 0, 255).astype(np.uint8)

            # 转置
            img = img.transpose((1, 2, 0))

            # 可视化
            if visualization:
                # 检查输出目录
                check_dir(self.output_dir)

                # 写入输出图片
                cv2.imwrite(os.path.join(self.output_dir, '%d_%d.jpg' % (im_id, time.time())), img)

            results.append(img)

        # 返回结果
        return results
