from paddlehub.common.logger import logger

OOV = -1


class WordCount(object):
    def __init__(self, word_id, count):
        self.word_id = word_id
        self.count = count


class Vocab(object):
    def __init__(self):
        self.__term2id = {}
        self.__id2term = {}

    def get_id(self, word):
        if word not in self.__term2id:
            return OOV
        return self.__term2id[word]

    def load(self, vocab_file):
        self.__term2id = {}
        self.__id2term = {}
        with open(vocab_file, 'r', encoding='utf-8') as fin:
            for line in fin.readlines():
                fields = line.strip().split('\t')
                assert len(fields) == 5, "Vocabulary file [%s] format error!" % (vocab_file)
                term = fields[1]
                id_ = int(fields[2])
                if term in self.__term2id:
                    logger.error("Duplicate word [%s] in vocab file!" % (term))
                    continue
                self.__term2id[term] = id_
                self.__id2term[id_] = term

    def size(self):
        return len(self.__term2id)

    def vocabulary(self):
        return self.__id2term
