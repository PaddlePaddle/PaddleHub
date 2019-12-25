import numpy as np


class BaseReader(object):
    def __init__(self, dataset, random_seed=None):
        self.dataset = dataset
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        np.random.seed(random_seed)

    def get_train_examples(self):
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        return self.dataset.get_dev_examples()

    def get_test_examples(self):
        return self.dataset.get_test_examples()

    def data_generator(self):
        raise NotImplementedError
