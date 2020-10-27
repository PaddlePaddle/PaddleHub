"""
This file defines the basic config information of LDA/SLDA model.
"""


class ModelType:
    LDA = 0
    SLDA = 1


class ModelConfig:
    type = None
    num_topics = None
    alpha = None
    beta = None
    word_topic_file = None
    vocab_file = None
