import os
import cv2
import numpy as np

__all__ = ['Processor']

class Processor():
    def __init__(self, paths, images, batch_size, input_size):
        # 图像列表
        self.imgs = self.load_datas(paths, images)

        # 输入数据
        self.input_datas = self.preprocess(self.imgs, batch_size, input_size)

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
    def preprocess(self, imgs, batch_size=1, input_size=320):
        input_datas = []
        for image in imgs:
            # h, w = image.shape[:2]
            # if h > w:
            #     new_h, new_w = input_size*h/w,input_size
            # else:
            #     new_h, new_w = input_size,input_size*w/h
            # image = cv2.resize(image, (int(new_w), int(new_h)))

            image = cv2.resize(image, (input_size, input_size))
            tmpImg = np.zeros((image.shape[0],image.shape[1],3))
            image = image/np.max(image)

            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

            # convert BGR to RGB
            tmpImg = tmpImg.transpose((2, 0, 1))
            tmpImg = tmpImg[np.newaxis,:,:,:]
            input_datas.append(tmpImg)
        
        input_datas = np.concatenate(input_datas, 0)

        datas_num = input_datas.shape[0]
        split_num = datas_num//batch_size+1 if datas_num%batch_size!=0 else datas_num//batch_size

        input_datas = np.array_split(input_datas, split_num)

        return input_datas

    def normPRED(self, d):
        ma = np.max(d)
        mi = np.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    # 后处理
    def postprocess(self, outputs, visualization=False, output_dir='output'):
        results = []
        if visualization and not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        for i, image in enumerate(self.imgs):
            # normalization
            pred = 1.0 - outputs[i,0,:,:]

            pred = self.normPRED(pred)

            # convert torch tensor to numpy array
            pred = pred.squeeze()
            pred = (pred*255).astype(np.uint8)
            h, w = image.shape[:2]
            pred = cv2.resize(pred, (w, h))

            results.append(pred)

            if visualization:
                cv2.imwrite(os.path.join(output_dir, 'result_%d.png' % i), pred)

        return results