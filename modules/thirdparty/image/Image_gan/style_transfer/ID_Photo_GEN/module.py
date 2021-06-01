import os
import cv2
import math
import paddle
import numpy as np
import paddle.nn as nn
import paddlehub as hub
from paddlehub.module.module import moduleinfo


@moduleinfo(
    name="ID_Photo_GEN",  # 模型名称
    type="CV",  # 模型类型
    author="jm12138",  # 作者名称
    author_email="jm12138@qq.com",  # 作者邮箱
    summary="ID_Photo_GEN",  # 模型介绍
    version="1.0.0"  # 版本号
)
class ID_Photo_GEN(nn.Layer):
    def __init__(self):
        super(ID_Photo_GEN, self).__init__()
        # 加载人脸关键点检测模型
        self.face_detector = hub.Module(name="face_landmark_localization")

        # 加载人脸分割模型
        self.seg = hub.Module(name='FCN_HRNet_W18_Face_Seg')

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
    def preprocess(self, images, batch_size, use_gpu):
        # 获取人脸关键点
        outputs = self.face_detector.keypoint_detection(images=images, batch_size=batch_size, use_gpu=use_gpu)

        crops = []
        for output, image in zip(outputs, images):
            for landmarks in output['data']:
                landmarks = np.array(landmarks)

                # rotation angle
                left_eye_corner = landmarks[36]
                right_eye_corner = landmarks[45]
                radian = np.arctan(
                    (left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))

                # image size after rotating
                height, width, _ = image.shape
                cos = math.cos(radian)
                sin = math.sin(radian)
                new_w = int(width * abs(cos) + height * abs(sin))
                new_h = int(width * abs(sin) + height * abs(cos))

                # translation
                Tx = new_w // 2 - width // 2
                Ty = new_h // 2 - height // 2

                # affine matrix
                M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx],
                              [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])

                image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))

                landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
                landmarks = np.dot(M, landmarks.T).T
                landmarks_top = np.min(landmarks[:, 1])
                landmarks_bottom = np.max(landmarks[:, 1])
                landmarks_left = np.min(landmarks[:, 0])
                landmarks_right = np.max(landmarks[:, 0])

                # expand bbox
                top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
                bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
                left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
                right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))

                # crop
                if bottom - top > right - left:
                    left -= ((bottom - top) - (right - left)) // 2
                    right = left + (bottom - top)
                else:
                    top -= ((right - left) - (bottom - top)) // 2
                    bottom = top + (right - left)

                image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255

                h, w = image.shape[:2]
                left_white = max(0, -left)
                left = max(0, left)
                right = min(right, w - 1)
                right_white = left_white + (right - left)
                top_white = max(0, -top)
                top = max(0, top)
                bottom = min(bottom, h - 1)
                bottom_white = top_white + (bottom - top)

                image_crop[top_white:bottom_white + 1, left_white:right_white + 1] = image[top:bottom + 1, left:right +
                                                                                           1].copy()
                crops.append(image_crop)

        # 获取人像分割的输出
        results = self.seg.Segmentation(images=crops, batch_size=batch_size)

        faces = []
        masks = []

        for result in results:
            # 提取MASK和输出图像
            face = result['face']
            mask = result['mask']

            faces.append(face)
            masks.append(mask)

        return faces, masks

    # 模型预测函数
    def predict(self, input_datas):
        outputs = []

        for data in input_datas:
            # 转换数据为Tensor
            data = paddle.to_tensor(data)

            # 模型前向计算
            cartoon = self.net(data)

            outputs.append(cartoon[0].numpy())

        outputs = np.concatenate(outputs, 0)

        return outputs

    # 结果后处理函数
    @staticmethod
    def postprocess(faces, masks, visualization, output_dir):
        # 检查输出目录
        if visualization:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        results = []

        for face, mask, i in zip(faces, masks, range(len(masks))):
            mask = mask[..., np.newaxis] / 255
            write = face * mask + (1 - mask) * 255
            blue = face * mask + (1 - mask) * [255, 0, 0]
            red = face * mask + (1 - mask) * [0, 0, 255]

            # 可视化结果保存
            if visualization:
                cv2.imwrite(os.path.join(output_dir, 'write_%d.jpg' % i), write)
                cv2.imwrite(os.path.join(output_dir, 'blue_%d.jpg' % i), blue)
                cv2.imwrite(os.path.join(output_dir, 'red_%d.jpg' % i), red)

            results.append({'write': write, 'blue': blue, 'red': red})

        return results

    def Photo_GEN(self, images=None, paths=None, batch_size=1, output_dir='output', visualization=False, use_gpu=False):

        # 获取输入数据
        images = self.load_datas(paths, images)

        # 数据预处理
        faces, masks = self.preprocess(images, batch_size, use_gpu)

        # 结果后处理
        results = self.postprocess(faces, masks, visualization, output_dir)

        return results
