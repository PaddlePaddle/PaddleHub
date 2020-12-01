import time
import yaml

import numpy as np
from paddlehub.common.logger import logger

from slda_novel.config import ModelType


def load_prototxt(config_file, config):
    """
    Args:
        config_file: model configuration file.
        config: ModelConfig class
    """
    logger.info("Loading SLDA config.")
    with open(config_file, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Assignment.
    if yaml_dict["type"] == "LDA":
        config.type = ModelType.LDA
    else:
        config.type = ModelType.SLDA
    config.num_topics = yaml_dict["num_topics"]
    config.alpha = yaml_dict["alpha"]
    config.beta = yaml_dict["beta"]
    config.word_topic_file = yaml_dict["word_topic_file"]
    config.vocab_file = yaml_dict["vocab_file"]


def fix_random_seed(seed=2147483647):
    np.random.seed(seed)


def rand(min_=0, max_=1):
    return np.random.uniform(low=min_, high=max_)


def rand_k(k):
    """Returns an integer float number between [0, k - 1].
    """
    return int(rand() * k)


def timeit(f):
    """Return time cost of function f.
    """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed
