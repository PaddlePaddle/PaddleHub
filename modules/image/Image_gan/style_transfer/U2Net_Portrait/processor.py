import os
import cv2
import numpy as np
import paddlehub as hub

__all__ = ['Processor']


class Processor():
    def __init__(self, paths, images, batch_size, face_detection=True, scale=1):
        # 图像列表
        self.imgs = self.load_datas(paths, images)

        # 输入数据
        self.input_datas = self.preprocess(self.imgs, batch_size, face_detection, scale)

    # 读取数据函数
    def load_datas(self, paths, images):
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

    # 预处理
    def preprocess(self, imgs, batch_size=1, face_detection=True, scale=1):
        if face_detection:
            # face detection
            face_detector = hub.Module(name="pyramidbox_lite_mobile")
            results = face_detector.face_detection(images=imgs, use_gpu=False, visualization=False, confs_threshold=0.5)
            im_faces = []
            for datas, img in zip(results, imgs):
                for face in datas['data']:
                    # get detection result
                    l, r, t, b = [face['left'], face['right'], face['top'], face['bottom']]

                    # square crop
                    pad = max(int(scale * (r - l)), int(scale * (b - t)))
                    c_w, c_h = (r - l) // 2 + l, (b - t) // 2 + t
                    top = 0 if c_h - pad < 0 else c_h - pad
                    bottom = pad + c_h
                    left = 0 if c_w - pad < 0 else c_w - pad
                    right = pad + c_w
                    crop = img[top:bottom, left:right]

                    # resize
                    im_face = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
                    im_faces.append(im_face)
        else:
            im_faces = []
            for img in imgs:
                h, w = img.shape[:2]
                if h > w:
                    if (h - w) % 2 == 0:
                        img = np.pad(
                            img, ((0, 0), ((h - w) // 2, (h - w) // 2), (0, 0)),
                            mode='constant',
                            constant_values=((255, 255), (255, 255), (255, 255)))
                    else:
                        img = np.pad(
                            img, ((0, 0), ((h - w) // 2, (h - w) // 2 + 1), (0, 0)),
                            mode='constant',
                            constant_values=((255, 255), (255, 255), (255, 255)))
                else:
                    if (w - h) % 2 == 0:
                        img = np.pad(
                            img, (((w - h) // 2, (w - h) // 2), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=((255, 255), (255, 255), (255, 255)))
                    else:
                        img = np.pad(
                            img, (((w - h) // 2, (w - h) // 2 + 1), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=((255, 255), (255, 255), (255, 255)))
                im_face = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                im_faces.append(im_face)

        input_datas = []
        for im_face in im_faces:
            tmpImg = np.zeros((im_face.shape[0], im_face.shape[1], 3))
            im_face = im_face / np.max(im_face)

            tmpImg[:, :, 0] = (im_face[:, :, 2] - 0.406) / 0.225
            tmpImg[:, :, 1] = (im_face[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (im_face[:, :, 0] - 0.485) / 0.229

            # convert BGR to RGB
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = tmpImg[np.newaxis, :, :, :]
            input_datas.append(tmpImg)

        input_datas = np.concatenate(input_datas, 0)

        datas_num = input_datas.shape[0]
        split_num = datas_num // batch_size + 1 if datas_num % batch_size != 0 else datas_num // batch_size

        input_datas = np.array_split(input_datas, split_num)

        return input_datas

    def normPRED(self, d):
        ma = np.max(d)
        mi = np.min(d)

        dn = (d - mi) / (ma - mi)

        return dn

    # 后处理
    def postprocess(self, outputs, visualization=False, output_dir='output'):
        results = []
        if visualization and not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for i in range(outputs.shape[0]):
            # normalization
            pred = 1.0 - outputs[i, 0, :, :]

            pred = self.normPRED(pred)

            # convert torch tensor to numpy array
            pred = pred.squeeze()
            pred = (pred * 255).astype(np.uint8)

            results.append(pred)

            if visualization:
                cv2.imwrite(os.path.join(output_dir, 'result_%d.png' % i), pred)

        return results
