import os

import paddlehub as hub
from paddlehub.module.module import moduleinfo
from paddlehub.common.logger import logger

from lda_novel.inference_engine import InferenceEngine
from lda_novel.document import LDADoc, SLDADoc
from lda_novel.semantic_matching import SemanticMatching, WordAndDis
from lda_novel.tokenizer import LACTokenizer, SimpleTokenizer
from lda_novel.config import ModelType
from lda_novel.vocab import Vocab, WordCount


@moduleinfo(
    name="lda_novel",
    version="1.0.2",
    summary=
    "This is a PaddleHub Module for LDA topic model in novel dataset, where we can calculate doc distance, calculate the similarity between query and document, etc.",
    author="DesmonDay",
    author_email="",
    type="nlp/semantic_model")
class TopicModel(hub.Module):
    def _initialize(self):
        """
        Initialize with the necessary elements.
        """
        self.model_dir = os.path.join(self.directory, 'novel')
        self.conf_file = 'lda.conf'
        self.__engine = InferenceEngine(self.model_dir, self.conf_file)
        self.vocab_path = os.path.join(self.model_dir, 'vocab_info.txt')
        lac = hub.Module(name="lac")
        # self.__tokenizer = SimpleTokenizer(self.vocab_path)
        self.__tokenizer = LACTokenizer(self.vocab_path, lac)

        self.vocabulary = self.__engine.get_model().get_vocab()
        self.config = self.__engine.get_config()
        self.topic_words = self.__engine.get_model().topic_words()
        self.topic_sum_table = self.__engine.get_model().topic_sum()

        def take_elem(word_count):
            return word_count.count

        for i in range(self.config.num_topics):
            self.topic_words[i].sort(key=take_elem, reverse=True)

        logger.info("Finish initialization.")

    def cal_doc_distance(self, doc_text1, doc_text2):
        """
        This interface calculates the distance between documents.

        Args:
            doc_text1(str): the input document text 1.
            doc_text2(str): the input document text 2.

        Returns:
            jsd(float): Jensen-Shannon Divergence distance of two documents.
            hd(float): Hellinger Distance of two documents.
        """
        doc1_tokens = self.__tokenizer.tokenize(doc_text1)
        doc2_tokens = self.__tokenizer.tokenize(doc_text2)

        # Document topic inference.
        doc1, doc2 = LDADoc(), LDADoc()
        self.__engine.infer(doc1_tokens, doc1)
        self.__engine.infer(doc2_tokens, doc2)

        # To calculate jsd, we need dense document topic distribution.
        dense_dict1 = doc1.dense_topic_dist()
        dense_dict2 = doc2.dense_topic_dist()
        # Calculate the distance between distributions.
        # The smaller the distance, the higher the document semantic similarity.
        sm = SemanticMatching()
        jsd = sm.jensen_shannon_divergence(dense_dict1, dense_dict2)
        hd = sm.hellinger_distance(dense_dict1, dense_dict2)

        return jsd, hd

    def cal_doc_keywords_similarity(self, document, top_k=10):
        """
        This interface can be used to find topk keywords of document.

        Args:
            document(str): the input document text.
            top_k(int): top k keywords of this document.

        Returns:
            results(list): contains top_k keywords and their corresponding
                           similarity compared to document.
        """
        d_tokens = self.__tokenizer.tokenize(document)

        # Do topic inference on documents to obtain topic distribution.
        doc = LDADoc()
        self.__engine.infer(d_tokens, doc)
        doc_topic_dist = doc.sparse_topic_dist()

        items = []
        words = set()
        for word in d_tokens:
            if word in words:
                continue
            words.add(word)
            wd = WordAndDis()
            wd.word = word
            sm = SemanticMatching()
            wd.distance = sm.likelihood_based_similarity(
                terms=[word], doc_topic_dist=doc_topic_dist, model=self.__engine.get_model())
            items.append(wd)

        def take_elem(word_dis):
            return word_dis.distance

        items.sort(key=take_elem, reverse=True)

        results = []
        size = len(items)
        for i in range(top_k):
            if i >= size:
                break
            results.append({"word": items[i].word, "similarity": items[i].distance})

        return results

    def cal_query_doc_similarity(self, query, document):
        """
        This interface calculates the similarity between query and document.

        Args:
            query(str): the input query text.
            document(str): the input document text.

        Returns:
            lda_sim(float): likelihood based similarity between query and document
                            based on LDA.
        """
        q_tokens = self.__tokenizer.tokenize(query)
        d_tokens = self.__tokenizer.tokenize(document)

        doc = LDADoc()
        self.__engine.infer(d_tokens, doc)
        doc_topic_dist = doc.sparse_topic_dist()

        sm = SemanticMatching()
        lda_sim = sm.likelihood_based_similarity(q_tokens, doc_topic_dist, self.__engine.get_model())

        return lda_sim

    def infer_doc_topic_distribution(self, document):
        """
        This interface infers the topic distribution of document.

        Args:
            document(str): the input document text.

        Returns:
            results(list): returns the topic distribution of document.
        """
        tokens = self.__tokenizer.tokenize(document)
        if tokens == []:
            return []
        results = []
        doc = LDADoc()
        self.__engine.infer(tokens, doc)
        topics = doc.sparse_topic_dist()
        for topic in topics:
            results.append({"topic id": topic.tid, "distribution": topic.prob})
        return results

    def show_topic_keywords(self, topic_id, k=10):
        """
        This interface returns the k keywords under specific topic.

        Args:
            topic_id(int): topic information we want to know.
            k(int): top k keywords.

        Returns:
            results(dict): contains specific topic's keywords and corresponding
                           probability.
        """
        EPS = 1e-8
        results = {}
        if 0 <= topic_id < self.config.num_topics:
            k = min(k, len(self.topic_words[topic_id]))
            for i in range(k):
                prob = self.topic_words[topic_id][i].count / \
                       (self.topic_sum_table[topic_id] + EPS)
                results[self.vocabulary[self.topic_words[topic_id][i].word_id]] = prob
            return results
        else:
            logger.error("%d is out of range!" % topic_id)
