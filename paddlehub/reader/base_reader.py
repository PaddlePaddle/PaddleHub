import numpy as np


class BaseReader(object):
    def __init__(self, dataset, random_seed=None):
        self.dataset = dataset
        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_train_examples(self):
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        return self.dataset.get_dev_examples()

    def get_test_examples(self):
        return self.dataset.get_test_examples()

    def data_generator(self):
        raise NotImplementedError
