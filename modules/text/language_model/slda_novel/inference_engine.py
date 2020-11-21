import os

from paddlehub.common.logger import logger

from slda_novel.config import ModelConfig
from slda_novel.util import load_prototxt, fix_random_seed, rand_k
from slda_novel.model import TopicModel
from slda_novel.sampler import GibbsSampler, MHSampler
from slda_novel.document import LDADoc, SLDADoc, Token, Sentence
from slda_novel.vocab import OOV


class SamplerType:
    GibbsSampling = 0
    MetropolisHastings = 1


class InferenceEngine(object):
    def __init__(self, model_dir, conf_file, type=SamplerType.MetropolisHastings):
        # Read model configuration.
        config = ModelConfig()
        conf_file_path = os.path.join(model_dir, conf_file)
        load_prototxt(conf_file_path, config)
        self.__model = TopicModel(model_dir, config)
        self.__config = config

        # Initialize the sampler according to the configuration.
        if type == SamplerType.GibbsSampling:
            self.__sampler = GibbsSampler(self.__model)
        elif type == SamplerType.MetropolisHastings:
            self.__sampler = MHSampler(self.__model)

    def infer(self, input, doc):
        """Perform LDA topic inference on input, and store the results in doc.
        Args:
            input: a list of strings after tokenization.
            doc: LDADoc type or SLDADoc type.
        """
        fix_random_seed()
        if isinstance(doc, LDADoc) and not isinstance(doc, SLDADoc):
            doc.init(self.__model.num_topics())
            doc.set_alpha(self.__model.alpha())
            for token in input:
                id_ = self.__model.term_id(token)
                if id_ != OOV:
                    init_topic = rand_k(self.__model.num_topics())
                    doc.add_token(Token(init_topic, id_))
            self.lda_infer(doc, 20, 50)
        elif isinstance(doc, SLDADoc):
            doc.init(self.__model.num_topics())
            doc.set_alpha(self.__model.alpha())
            for sent in input:
                words = []
                for token in sent:
                    id_ = self.__model.term_id(token)
                    if id_ != OOV:
                        words.append(id_)
                init_topic = rand_k(self.__model.num_topics())
                doc.add_sentence(Sentence(init_topic, words))
            self.slda_infer(doc, 20, 50)
        else:
            logger.error("Wrong Doc Type!")

    def lda_infer(self, doc, burn_in_iter, total_iter):
        assert burn_in_iter >= 0
        assert total_iter > 0
        assert total_iter > burn_in_iter

        for iter_ in range(total_iter):
            self.__sampler.sample_doc(doc)
            if iter_ >= burn_in_iter:
                doc.accumulate_topic_num()

    def slda_infer(self, doc, burn_in_iter, total_iter):
        assert burn_in_iter >= 0
        assert total_iter > 0
        assert total_iter > burn_in_iter

        for iter_ in range(total_iter):
            self.__sampler.sample_doc(doc)
            if iter_ >= burn_in_iter:
                doc.accumulate_topic_num()

    def model_type(self):
        return self.__model.type()

    def get_model(self):
        return self.__model

    def get_config(self):
        return self.__config
