import numpy as np

from paddlehub.common.logger import logger


class BaseReader(object):
    def __init__(self, dataset, random_seed=None, label_list=None):
        self.dataset = dataset
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}
        np.random.seed(random_seed)

        # generate label map
        self.label_map = {}

        try:
            for index, label in enumerate(self.dataset.get_labels()):
                self.label_map[label] = index
        except:
            # some dataset like squad, its label_list=None
            logger.info("Dataset is None or it has not any labels")
            if label_list:
                logger.info("Label list has been set")
                for index, label in enumerate(label_list):
                    self.label_map[label] = index

        logger.info("Reader label map = {}".format(self.label_map))

    def get_train_examples(self):
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        return self.dataset.get_dev_examples()

    def get_test_examples(self):
        return self.dataset.get_test_examples()

    def data_generator(self):
        raise NotImplementedError
