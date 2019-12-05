import numpy as np


class BasicReader(object):
    def __init__(self, dataset, random_seed=None):
        self.dataset = dataset
        np.random.seed(random_seed)

        self.current_example = 0
        self.current_epoch = 0
        self.num_examples = {'train': -1, 'dev': -1, 'test': -1}

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        return self.dataset.get_train_examples()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        return self.dataset.get_dev_examples()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for prediction."""
        return self.dataset.get_test_examples()

    def get_train_progress(self):
        """Gets progress for training phase."""
        return self.current_example, self.current_epoch

    def data_generator(self):
        raise NotImplementedError

    def get_num_examples(self, phase):
        """Get number of examples for train, dev or test."""
        if phase not in ['train', 'val', 'dev', 'test']:
            raise ValueError(
                "Unknown phase, which should be in ['train', 'val'/'dev', 'test']."
            )
        return self.num_examples[phase]
