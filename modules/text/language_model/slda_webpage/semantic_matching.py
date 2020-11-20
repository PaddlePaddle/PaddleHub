import os

import numpy as np
from paddlehub.common.logger import logger

from slda_webpage.vocab import OOV

EPS = 1e-06


class WordAndDis(object):
    def __init__(self):
        self.word = None
        self.distance = None


class SemanticMatching(object):
    def __init__(self):
        pass

    def l2_norm(self, vec):
        """Calculate the length of vector.
        """
        result = np.sqrt(np.sum(vec**2))
        return result

    def cosine_similarity(self, vec1, vec2):
        norm1 = self.l2_norm(vec1)
        norm2 = self.l2_norm(vec2)
        result = np.sum(vec1 * vec2) / norm1 / norm2
        return result

    def likelihood_based_similarity(self, terms, doc_topic_dist, model):
        """
        Args:
            terms: list of strings
            doc_topic_dist: list of Topic class
            model: TopicModel class
        """
        num_of_term_in_vocab = 0
        result = 0
        for i in range(len(terms)):
            term_id = model.term_id(terms[i])
            if term_id == OOV:
                continue
            num_of_term_in_vocab += 1
            for j in range(len(doc_topic_dist)):
                topic_id = doc_topic_dist[j].tid
                prob = doc_topic_dist[j].prob
                result += model.word_topic_value(term_id, topic_id) * 1.0 / \
                          model.topic_sum_value(topic_id) * prob

        if num_of_term_in_vocab == 0:
            return result
        return result / num_of_term_in_vocab

    def kullback_leibler_divergence(self, dist1, dist2):
        assert dist1.shape == dist2.shape
        dist2[dist2 < EPS] = EPS
        result = np.sum(dist1 * np.log(dist1 / dist2))
        return result

    def jensen_shannon_divergence(self, dist1, dist2):
        assert dist1.shape == dist2.shape
        dist1[dist1 < EPS] = EPS
        dist2[dist2 < EPS] = EPS
        mean = (dist1 + dist2) * 0.5
        jsd = self.kullback_leibler_divergence(dist1, mean) * 0.5 + \
              self.kullback_leibler_divergence(dist2, mean) * 0.5
        return jsd

    def hellinger_distance(self, dist1, dist2):
        assert dist1.shape == dist2.shape
        result = np.sum((np.sqrt(dist1) - np.sqrt(dist2))**2)
        result = np.sqrt(result) * 0.7071067812
        return result
