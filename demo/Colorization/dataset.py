import os

import paddle.fluid as fluid


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in dirnames:
            img_path = os.path.join(parent,filename)
            if os.path.exists(img_path):
                for file in os.listdir(img_path):
                    if file.lower().endswith (('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                        imagelist.append(os.path.join(img_path, file))
    imagelist.sort()
    return imagelist


class Colorization(fluid.io.Dataset):
    def __init__(self, transform, mode='train'):
        self.mode = mode
        self.transform = transform

        if self.mode == 'train':
            self.file = '/Users/haoyuying/Downloads/超分数据集/test/SRF_2'
        elif self.mode == 'test':
            self.file = '/Users/haoyuying/Downloads/超分数据集/test/SRF_2'
        else:
            self.file = '/Users/haoyuying/Downloads/超分数据集/test/SRF_2'
        self.data = get_img_file(self.file)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        im = self.transform(img_path)
        return im['A'], im['hint_B'], im['mask_B'], im['B'], im['real_B_enc']

    def __len__(self):
        return len(self.data)



