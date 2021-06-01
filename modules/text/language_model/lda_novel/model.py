import os
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
from paddlehub.common.logger import logger

from lda_novel.vocab import Vocab, WordCount


class TopicModel(object):
    """Storage Structure of Topic model, including vocabulary and word topic count.
    """

    def __init__(self, model_dir, config):
        """
        Args:
            model_dir: the path of model directory
            config: ModelConfig class.
        """
        self.__word_topic = None  # Model parameter of word topic.
        self.__vocab = Vocab()  # Vocab data structure of model.
        self.__num_topics = config.num_topics  # Number of topics.
        self.__alpha = config.alpha
        self.__alpha_sum = self.__alpha * self.__num_topics
        self.__beta = config.beta
        self.__beta_sum = None
        self.__type = config.type  # Model type.
        self.__topic_sum = np.zeros(self.__num_topics, dtype="int64")  # Accum sum of each topic in word topic.
        self.__topic_words = [[] for _ in range(self.__num_topics)]
        word_topic_path = os.path.join(model_dir, config.word_topic_file)
        vocab_path = os.path.join(model_dir, config.vocab_file)
        self.load_model(word_topic_path, vocab_path)

    def term_id(self, term):
        return self.__vocab.get_id(term)

    def load_model(self, word_topic_path, vocab_path):

        # Loading vocabulary
        self.__vocab.load(vocab_path)

        self.__beta_sum = self.__beta * self.__vocab.size()
        self.__word_topic = [{} for _ in range(self.__vocab.size())]  # 字典列表
        self.__load_word_dict(word_topic_path)
        logger.info("Model Info: #num_topics=%d #vocab_size=%d alpha=%f beta=%f" %
                    (self.num_topics(), self.vocab_size(), self.alpha(), self.beta()))

    def word_topic_value(self, word_id, topic_id):
        """Return value of specific word under specific topic in the model.
        """
        word_dict = self.__word_topic[word_id]
        if topic_id not in word_dict:
            return 0
        return word_dict[topic_id]

    def word_topic(self, term_id):
        """Return the topic distribution of a word.
        """
        return self.__word_topic[term_id]

    def topic_sum_value(self, topic_id):
        return self.__topic_sum[topic_id]

    def topic_sum(self):
        return self.__topic_sum

    def num_topics(self):
        return self.__num_topics

    def vocab_size(self):
        return self.__vocab.size()

    def alpha(self):
        return self.__alpha

    def alpha_sum(self):
        return self.__alpha_sum

    def beta(self):
        return self.__beta

    def beta_sum(self):
        return self.__beta_sum

    def type(self):
        return self.__type

    def __load_word_dict(self, word_dict_path):
        """Load the word topic parameters.
        """
        logger.info("Loading word topic.")
        with open(word_dict_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                fields = line.strip().split(" ")
                assert len(fields) > 0, "Model file format error!"
                term_id = int(fields[0])
                assert term_id < self.vocab_size(), "Term id out of range!"
                assert term_id >= 0, "Term id out of range!"
                for i in range(1, len(fields)):
                    topic_count = fields[i].split(":")
                    assert len(topic_count) == 2, "Topic count format error!"

                    topic_id = int(topic_count[0])
                    assert topic_id >= 0, "Topic out of range!"
                    assert topic_id < self.__num_topics, "Topic out of range!"

                    count = int(topic_count[1])
                    assert count >= 0, "Topic count error!"

                    self.__word_topic[term_id][topic_id] = count
                    self.__topic_sum[topic_id] += count
                    self.__topic_words[topic_id].append(WordCount(term_id, count))
                new_dict = OrderedDict()
                for key in sorted(self.__word_topic[term_id]):
                    new_dict[key] = self.__word_topic[term_id][key]
                self.__word_topic[term_id] = new_dict

    def get_vocab(self):
        return self.__vocab.vocabulary()

    def topic_words(self):
        return self.__topic_words
