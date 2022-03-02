import paddle
import numpy as np
from typing import Callable
from code.config import config_parameters


class GemStones(paddle.io.Dataset):
    """
    step 1：paddle.io.Dataset
    """

    def __init__(self, transforms: Callable, mode: str = 'train'):
        """
        step 2：create reader
        """
        super(GemStones, self).__init__()

        self.mode = mode
        self.transforms = transforms

        train_image_dir = config_parameters['train_image_dir']
        eval_image_dir = config_parameters['eval_image_dir']
        test_image_dir = config_parameters['test_image_dir']

        train_data_folder = paddle.vision.DatasetFolder(train_image_dir)
        eval_data_folder = paddle.vision.DatasetFolder(eval_image_dir)
        test_data_folder = paddle.vision.DatasetFolder(test_image_dir)

        config_parameters['label_dict'] = train_data_folder.class_to_idx

        if self.mode == 'train':
            self.data = train_data_folder
        elif self.mode == 'eval':
            self.data = eval_data_folder
        elif self.mode == 'test':
            self.data = test_data_folder

    def __getitem__(self, index):
        """
        step 3：implement __getitem__
        """
        data = np.array(self.data[index][0]).astype('float32')

        data = self.transforms(data)

        label = np.array(self.data[index][1]).astype('int64')

        return data, label

    def __len__(self):
        """
        step 4：implement __len__
        """
        return len(self.data)
