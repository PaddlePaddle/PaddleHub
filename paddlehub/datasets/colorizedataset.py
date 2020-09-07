import os

import numpy
import paddle


class Colorizedataset(paddle.io.Dataset):
    """
    Dataset for colorization.
    Args:
       transform(callmethod) : The method of preprocess images.
       mode(str): The mode for preparing dataset.
    Returns:
        DataSet: An iterable object for data iterating
    """
    def __init__(self, transform: classmethod, mode: str = 'train'):
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.file = '/PATH/FOR/TRAIN/SET'
        elif self.mode == 'test':
            self.file = '/PATH/FOR/TEST/SET'
        else:
            self.file = '/PATH/FOR/VALIDATION'
        self.data = get_img_file(self.file)

    def __getitem__(self, idx: int) -> numpy.ndarray:
        img_path = self.data[idx]
        im = self.transform(img_path)
        return im['A'], im['hint_B'], im['mask_B'], im['B'], im['real_B_enc']

    def __len__(self):
        return len(self.data)