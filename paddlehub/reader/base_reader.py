import numpy as np

from paddlehub.common.logger import logger


class BaseReader(object):
    def __init__(self, dataset, random_seed=None):
        self.dataset = dataset
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        np.random.seed(random_seed)

        # generate label map
        self.label_map = {}
        try:
            for index, label in enumerate(self.dataset.get_labels()):
                self.label_map[label] = index
            logger.info("Dataset label map = {}".format(self.label_map))
        except:
            # some dataset like squad, its label_list=None
            logger.info(
                "Dataset is None or it has not any labels, label map = {}".
                format(self.label_map))

    def get_train_examples(self):
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        return self.dataset.get_dev_examples()

    def get_test_examples(self):
        return self.dataset.get_test_examples()

    def data_generator(self):
        raise NotImplementedError
