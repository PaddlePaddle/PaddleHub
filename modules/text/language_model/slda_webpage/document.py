import numpy as np


class Topic(object):
    """Basic data structure of topic, contains topic id and
       corresponding probability.
    """

    def __init__(self, tid, prob):
        self.tid = tid  # topic id
        self.prob = prob  # topic probability


class Token(object):
    """Basic storage unit of LDA documents, contains word id
       and corresponding topic.
    """

    def __init__(self, topic, id):
        self.topic = topic
        self.id = id


class Sentence(object):
    """Basic storage unit of SentenceLDA documents, contains word ids
       of the sentence and its corresponding topic id.
    """

    def __init__(self, topic, tokens):
        self.topic = topic
        self.tokens = tokens


class LDADoc(object):
    """The storage structure of LDA model's inference result.
    """

    def __init__(self):
        self._num_topics = None  # Number of topics.
        self._num_accum = None  # Number of accumulated sample rounds.
        self._alpha = None  # Document prior parameter.
        self._tokens = None  # Storage structure of inference results.
        self._topic_sum = None  # Document's topic sum in one round samples.
        self._accum_topic_sum = None  # Accumulated results of topic sum.

    def init(self, num_topics):
        """Initialize the LDADoc according to num_topics.
        """
        self._num_topics = num_topics
        self._num_accum = 0
        self._tokens = []
        self._topic_sum = np.zeros(self._num_topics)
        self._accum_topic_sum = np.zeros(self._num_topics)

    def add_token(self, token):
        """Add new word to current LDADoc.
        Arg:
            token: Token class object.
        """
        assert token.topic >= 0, "Topic %d out of range!" % token.topic
        assert token.topic < self._num_topics, "Topic %d out of range!" % token.topic
        self._tokens.append(token)
        self._topic_sum[token.topic] += 1

    def token(self, index):
        return self._tokens[index]

    def set_topic(self, index, new_topic):
        """Set the index word's topic to new_topic, and update the corresponding
           topic distribution.
        """
        assert new_topic >= 0, "Topic %d out of range!" % new_topic
        assert new_topic < self._num_topics, "Topic %d out of range!" % new_topic
        old_topic = self._tokens[index].topic
        if new_topic == old_topic:
            return
        self._tokens[index].topic = new_topic
        self._topic_sum[old_topic] -= 1
        self._topic_sum[new_topic] += 1

    def set_alpha(self, alpha):
        self._alpha = alpha

    def size(self):
        """Return number of words in LDADoc.
        """
        return len(self._tokens)

    def topic_sum(self, topic_id):
        return self._topic_sum[topic_id]

    def sparse_topic_dist(self, sort=True):
        """Return the topic distribution of documents in sparse format.
           By default, it is sorted according to the topic probability
           under the descending order.
        """
        topic_dist = []
        sum_ = np.sum(self._accum_topic_sum)
        if sum_ == 0:
            return topic_dist
        for i in range(0, self._num_topics):
            if self._accum_topic_sum[i] == 0:
                continue
            topic_dist.append(Topic(i, self._accum_topic_sum[i] * 1.0 / sum_))
        if sort:

            def take_elem(topic):
                return topic.prob

            topic_dist.sort(key=take_elem, reverse=True)
            if topic_dist is None:
                topic_dist = []

        return topic_dist

    def dense_topic_dist(self):
        """Return the distribution of document topics in dense format,
           taking into account the prior parameter alpha.
        """
        dense_dist = np.zeros(self._num_topics)
        if self.size() == 0:
            return dense_dist
        dense_dist = (self._accum_topic_sum * 1.0 / self._num_accum + self._alpha) / (
            self.size() + self._alpha * self._num_topics)
        return dense_dist

    def accumulate_topic_num(self):
        self._accum_topic_sum += self._topic_sum
        self._num_accum += 1


class SLDADoc(LDADoc):
    """Sentence LDA Document, inherited from LDADoc.
       Add add_sentence interface.
    """

    def __init__(self):
        super().__init__()
        self.__sentences = None

    def init(self, num_topics):
        """Initialize the SLDADoc according to num_topics.
        """
        self._num_topics = num_topics
        self.__sentences = []
        self._num_accum = 0
        self._topic_sum = np.zeros(self._num_topics)
        self._accum_topic_sum = np.zeros(self._num_topics)

    def add_sentence(self, sent):
        """Add new sentence to current SLDADoc.
        Arg:
            sent: Sentence class object.
        """
        assert sent.topic >= 0, "Topic %d out of range!" % (sent.topic)
        assert sent.topic < self._num_topics, "Topic %d out of range!" % (sent.topic)
        self.__sentences.append(sent)
        self._topic_sum[sent.topic] += 1

    def set_topic(self, index, new_topic):
        assert new_topic >= 0, "Topic %d out of range!" % (new_topic)
        assert new_topic < self._num_topics, "Topic %d out of range!" % (new_topic)
        old_topic = self.__sentences[index].topic
        if new_topic == old_topic:
            return
        self.__sentences[index].topic = new_topic
        self._topic_sum[old_topic] -= 1
        self._topic_sum[new_topic] += 1

    def size(self):
        """Return number of sentences in SLDADoc.
        """
        return len(self.__sentences)

    def sent(self, index):
        return self.__sentences[index]
