import os
import cv2
import time
import base64
import numpy as np

__all__ = ['base64_to_cv2', 'Processor']


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
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


class Processor():
    # 初始化函数
    def __init__(self, images=None, paths=None, batch_size=1, output_dir='output'):
        # 变量设置
        self.num_points = 21
        self.inHeight = 368
        self.threshold = 0.1
        self.point_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
                            [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

        self.images = images
        self.paths = paths
        self.batch_size = batch_size
        self.output_dir = output_dir

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
                im = cv2.imread(im_path).astype('float32')
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
            img_height, img_width, _ = img.shape
            aspect_ratio = img_width / img_height
            inWidth = int(((aspect_ratio * self.inHeight) * 8) // 8)
            inpBlob = cv2.dnn.blobFromImage(
                img, 1.0 / 255, (inWidth, self.inHeight), (0, 0, 0), swapRB=False, crop=False)
            input_datas.append(inpBlob)

        # 数据按batch_size切分
        input_datas = np.concatenate(input_datas, 0)
        split_num = len(self.datas) // self.batch_size + 1 if len(self.datas) % self.batch_size != 0 else len(
            self.datas) // self.batch_size
        input_datas = np.array_split(input_datas, split_num)

        # 返回预处理完成的数据
        return input_datas

    # 结果后处理函数
    def postprocess(self, outputs, visualization):
        all_points = []

        # 结果后处理
        for im_id, img in enumerate(self.datas):
            points = []
            for idx in range(self.num_points):
                probMap = outputs[im_id, idx, :, :]
                img_height, img_width, _ = img.shape
                probMap = cv2.resize(probMap, (img_width, img_height))
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

                if prob > self.threshold:
                    points.append([int(point[0]), int(point[1])])
                else:
                    points.append(None)

            all_points.append(points)

            # 结果可视化
            if visualization:
                # 检查输出目录
                check_dir(self.output_dir)
                # 结果可视化
                self.vis_pose(img, points, im_id)

        # 返回后处理结果
        return all_points

    # 结果可视化
    def vis_pose(self, img, points, im_id):
        # 根据结果绘制关键点到原图像上
        for pair in self.point_pairs:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(img, tuple(points[partA]), tuple(points[partB]), (0, 255, 255), 3)
                cv2.circle(img, tuple(points[partA]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        # 可视化图像保存
        cv2.imwrite(os.path.join(self.output_dir, '%d_%d.jpg' % (im_id, time.time())), img)
