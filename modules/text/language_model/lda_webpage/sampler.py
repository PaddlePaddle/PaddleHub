import os

import numpy as np
from tqdm import tqdm
from paddlehub.common.logger import logger

from lda_webpage.document import LDADoc, SLDADoc, Token, Sentence
from lda_webpage.vose_alias import VoseAlias
from lda_webpage.util import rand, rand_k


class Sampler(object):
    def __init__(self):
        pass

    def sample_doc(self, doc):
        """Sample LDA or SLDA topics for documents.
        """
        raise NotImplementedError


class MHSampler(Sampler):
    def __init__(self, model):
        super().__init__()
        self.__model = model
        self.__topic_indexes = None
        self.__alias_tables = None
        self.__prob_sum = None
        self.__beta_alias = VoseAlias()
        self.__beta_prior_sum = None
        self.__mh_steps = 2
        self.__construct_alias_table()

    def __construct_alias_table(self):
        """Construct alias table for all words.
        """
        logger.info("Construct alias table for alias sampling method.")
        vocab_size = self.__model.vocab_size()
        self.__topic_indexes = [[] for _ in range(vocab_size)]
        self.__alias_tables = [VoseAlias() for _ in range(vocab_size)]
        self.__prob_sum = np.zeros(vocab_size)

        # Construct each word's alias table (prior is not included).
        for i in tqdm(range(vocab_size)):
            dist = []
            prob_sum = 0
            for key in self.__model.word_topic(i):
                topic_id = key
                word_topic_count = self.__model.word_topic(i)[key]
                topic_sum = self.__model.topic_sum_value(topic_id)

                self.__topic_indexes[i].append(topic_id)
                q = word_topic_count / (topic_sum + self.__model.beta_sum())
                dist.append(q)
                prob_sum += q
            self.__prob_sum[i] = prob_sum
            if len(dist) > 0:
                dist = np.array(dist, dtype=np.float)
                self.__alias_tables[i].initialize(dist)

        # Build prior parameter beta's alias table.
        beta_dist = self.__model.beta() / (self.__model.topic_sum() + self.__model.beta_sum())
        self.__beta_prior_sum = np.sum(beta_dist)
        self.__beta_alias.initialize(beta_dist)

    def sample_doc(self, doc):
        if isinstance(doc, LDADoc) and not isinstance(doc, SLDADoc):
            for i in range(doc.size()):
                new_topic = self.__sample_token(doc, doc.token(i))
                doc.set_topic(i, new_topic)
        elif isinstance(doc, SLDADoc):
            for i in range(doc.size()):
                new_topic = self.__sample_sentence(doc, doc.sent(i))
                doc.set_topic(i, new_topic)

    def __sample_token(self, doc, token):
        new_topic = token.topic
        for i in range(self.__mh_steps):
            doc_proposed_topic = self.__doc_proposal(doc, token)
            new_topic = self.__word_proposal(doc, token, doc_proposed_topic)
        return new_topic

    def __sample_sentence(self, doc, sent):
        new_topic = sent.topic
        for i in range(self.__mh_steps):
            doc_proposed_topic = self.__doc_proposal(doc, sent)
            new_topic = self.__word_proposal(doc, sent, doc_proposed_topic)
        return new_topic

    def __doc_proposal(self, doc, token):
        if isinstance(doc, LDADoc) and isinstance(token, Token):
            old_topic = token.topic
            dart = rand() * (doc.size() + self.__model.alpha_sum())
            if dart < doc.size():
                token_index = int(dart)
                new_topic = doc.token(token_index).topic
            else:
                new_topic = rand_k(self.__model.num_topics())

            if new_topic != old_topic:
                proposal_old = self.__doc_proposal_distribution(doc, old_topic)
                proposal_new = self.__doc_proposal_distribution(doc, new_topic)
                proportion_old = self.__proportional_function(doc, token, old_topic)
                proportion_new = self.__proportional_function(doc, token, new_topic)
                transition_prob = float((proportion_new * proposal_old) / (proportion_old * proposal_new))
                rejection = rand()
                mask = -(rejection < transition_prob)
                return (new_topic & mask) | (old_topic & ~mask)

            return new_topic

        elif isinstance(doc, SLDADoc) and isinstance(token, Sentence):
            sent = token
            old_topic = sent.topic
            dart = rand() * (doc.size() + self.__model.alpha_sum())
            if dart < doc.size():
                token_index = int(dart)
                new_topic = doc.sent(token_index).topic
            else:
                new_topic = rand_k(self.__model.num_topics())

            if new_topic != old_topic:
                proportion_old = self.__proportional_function(doc, sent, old_topic)
                proportion_new = self.__proportional_function(doc, sent, new_topic)
                proposal_old = self.__doc_proposal_distribution(doc, old_topic)
                proposal_new = self.__doc_proposal_distribution(doc, new_topic)
                transition_prob = float((proportion_new * proposal_old) / (proportion_old * proposal_new))
                rejection = rand()
                mask = -(rejection < transition_prob)
                return (new_topic & mask) | (old_topic & ~mask)

            return new_topic

    def __word_proposal(self, doc, token, old_topic):
        if isinstance(doc, LDADoc) and isinstance(token, Token):
            new_topic = self.__propose(token.id)
            if new_topic != old_topic:
                proposal_old = self.__word_proposal_distribution(token.id, old_topic)
                proposal_new = self.__word_proposal_distribution(token.id, new_topic)
                proportion_old = self.__proportional_function(doc, token, old_topic)
                proportion_new = self.__proportional_function(doc, token, new_topic)
                transition_prob = float((proportion_new * proposal_old) / (proportion_old * proposal_new))
                rejection = rand()
                mask = -(rejection < transition_prob)
                return (new_topic & mask) | (old_topic & ~mask)
            return new_topic

        elif isinstance(doc, SLDADoc) and isinstance(token, Sentence):
            sent = token
            new_topic = old_topic
            for word_id in sent.tokens:
                new_topic = self.__propose(word_id)
                if new_topic != old_topic:
                    proportion_old = self.__proportional_function(doc, sent, old_topic)
                    proportion_new = self.__proportional_function(doc, sent, new_topic)
                    proposal_old = self.__word_proposal_distribution(word_id, old_topic)
                    proposal_new = self.__word_proposal_distribution(word_id, new_topic)
                    transition_prob = float((proportion_new * proposal_old) / (proportion_old * proposal_new))
                    rejection = rand()
                    mask = -(rejection < transition_prob)
                    new_topic = (new_topic & mask) | (old_topic & ~mask)
            return new_topic

    def __proportional_function(self, doc, token, new_topic):
        if isinstance(doc, LDADoc) and isinstance(token, Token):
            old_topic = token.topic
            dt_alpha = doc.topic_sum(new_topic) + self.__model.alpha()
            wt_beta = self.__model.word_topic_value(token.id, new_topic) + self.__model.beta()
            t_sum_beta_sum = self.__model.topic_sum_value(new_topic) + self.__model.beta_sum()
            if new_topic == old_topic and wt_beta > 1:
                if dt_alpha > 1:
                    dt_alpha -= 1
                wt_beta -= 1
                t_sum_beta_sum -= 1
            return dt_alpha * wt_beta / t_sum_beta_sum

        elif isinstance(doc, SLDADoc) and isinstance(token, Sentence):
            sent = token
            old_topic = sent.topic
            result = doc.topic_sum(new_topic) + self.__model.alpha()
            if new_topic == old_topic:
                result -= 1
            for word_id in sent.tokens:
                wt_beta = self.__model.word_topic_value(word_id, new_topic) + self.__model.beta()
                t_sum_beta_sum = self.__model.topic_sum_value(new_topic) + self.__model.beta_sum()
                if new_topic == old_topic and wt_beta > 1:
                    wt_beta -= 1
                    t_sum_beta_sum -= 1
                result *= wt_beta / t_sum_beta_sum
            return result
        else:
            logger.error("Wrong input argument type!")

    def __word_proposal_distribution(self, word_id, topic):
        wt_beta = self.__model.word_topic_value(word_id, topic) + self.__model.beta()
        t_sum_beta_sum = self.__model.topic_sum_value(topic) + self.__model.beta_sum()
        return wt_beta / t_sum_beta_sum

    def __doc_proposal_distribution(self, doc, topic):
        return doc.topic_sum(topic) + self.__model.alpha()

    def __propose(self, word_id):
        dart = rand() * (self.__prob_sum[word_id] + self.__beta_prior_sum)
        if dart < self.__prob_sum[word_id]:
            idx = self.__alias_tables[word_id].generate()
            topic = self.__topic_indexes[word_id][idx]
        else:
            topic = self.__beta_alias.generate()
        return topic


class GibbsSampler(Sampler):
    def __init__(self, model):
        super().__init__()
        self.__model = model

    def sample_doc(self, doc):
        if isinstance(doc, LDADoc) and not isinstance(doc, SLDADoc):
            for i in range(doc.size()):
                new_topic = self.__sample_token(doc, doc.token(i))
                doc.set_topic(i, new_topic)
        elif isinstance(doc, SLDADoc):
            for i in range(doc.size()):
                new_topic = self.__sample_sentence(doc, doc.sent(i))
                doc.set_topic(i, new_topic)

    def __sample_token(self, doc, token):
        old_topic = token.topic
        num_topics = self.__model.num_topics()
        accum_prob = np.zeros(num_topics)
        prob = np.zeros(num_topics)
        sum_ = 0
        for i in range(num_topics):
            dt_alpha = doc.topic_sum(i) + self.__model.alpha()
            wt_beta = self.__model.word_topic_value(token.id, i) + self.__model.beta()
            t_sum_beta_sum = self.__model.topic_sum(i) + self.__model.beta_sum()
            if i == old_topic and wt_beta > 1:
                if dt_alpha > 1:
                    dt_alpha -= 1
                wt_beta -= 1
                t_sum_beta_sum -= 1
            prob[i] = dt_alpha * wt_beta / t_sum_beta_sum
            sum_ += prob[i]
            accum_prob[i] = prob[i] if i == 0 else accum_prob[i - 1] + prob[i]

        dart = rand() * sum_
        if dart <= accum_prob[0]:
            return 0
        for i in range(1, num_topics):
            if accum_prob[i - 1] < dart <= accum_prob[i]:
                return i
        return num_topics - 1

    def __sample_sentence(self, doc, sent):
        old_topic = sent.topic
        num_topics = self.__model.num_topics()
        accum_prob = np.zeros(num_topics)
        prob = np.zeros(num_topics)
        sum_ = 0
        for t in range(num_topics):
            dt_alpha = doc.topic_sum(t) + self.__model.alpha()
            t_sum_beta_sum = self.__model.topic_sum(t) + self.__model.beta_sum()
            if t == old_topic:
                if dt_alpha > 1:
                    dt_alpha -= 1
                if t_sum_beta_sum > 1:
                    t_sum_beta_sum -= 1
            prob[t] = dt_alpha
            for i in range(len(sent.tokens)):
                w = sent.tokens[i]
                wt_beta = self.__model.word_topic_value(w, t) + self.__model.beta()
                if t == old_topic and wt_beta > 1:
                    wt_beta -= 1
                # Note: if the length of the sentence is too long, the probability will be
                # too small and the accuracy will be lost if there are too many multiply items
                prob[t] *= wt_beta / t_sum_beta_sum
            sum_ += prob[t]
            accum_prob[t] = prob[t] if t == 0 else accum_prob[t - 1] + prob[t]

        dart = rand() * sum
        if dart <= accum_prob[0]:
            return 0
        for t in range(1, num_topics):
            if accum_prob[t - 1] < dart <= accum_prob[t]:
                return t
        return num_topics - 1
