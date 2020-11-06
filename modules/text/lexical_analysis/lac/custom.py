# -*- coding: UTF-8 -*-
"""
该模块实现用户自定义词典的功能
"""

from io import open

from .ahocorasick import Ahocorasick


class Customization(object):
    """
    基于AC自动机实现用户干预的功能
    """

    def __init__(self):
        self.dictitem = {}
        self.ac = None
        pass

    def load_customization(self, filename, sep=None):
        """装载人工干预词典"""
        self.ac = Ahocorasick()
        with open(filename, 'r', encoding='utf8') as f:
            for line in f:
                if sep == None:
                    words = line.strip().split()
                else:
                    words = line.strip().split(sep)

                if len(words) == 0:
                    continue

                phrase = ""
                tags = []
                offset = []
                for word in words:
                    if word.rfind('/') < 1:
                        phrase += word
                        tags.append('')
                    else:
                        phrase += word[:word.rfind('/')]
                        tags.append(word[word.rfind('/') + 1:])
                    offset.append(len(phrase))

                if len(phrase) < 2 and tags[0] == '':
                    continue

                self.dictitem[phrase] = (tags, offset)
                self.ac.add_word(phrase)
        self.ac.make()

    def parse_customization(self, query, lac_tags):
        """使用人工干预词典修正lac模型的输出"""

        def ac_postpress(ac_res):
            ac_res.sort()
            i = 1
            while i < len(ac_res):
                if ac_res[i - 1][0] < ac_res[i][0] and ac_res[i][0] <= ac_res[i - 1][1]:
                    ac_res.pop(i)
                    continue
                i += 1
            return ac_res

        if not self.ac:
            print("Customized dict is not loaded.")
            return

        ac_res = self.ac.search(query)

        ac_res = ac_postpress(ac_res)

        for begin, end in ac_res:
            phrase = query[begin:end + 1]
            index = begin

            tags, offsets = self.dictitem[phrase]
            for tag, offset in zip(tags, offsets):
                while index < begin + offset:
                    if len(tag) == 0:
                        lac_tags[index] = lac_tags[index][:-1] + 'I'
                    else:
                        lac_tags[index] = tag + "-I"
                    index += 1

            lac_tags[begin] = lac_tags[begin][:-1] + 'B'
            for offset in offsets:
                index = begin + offset
                if index < len(lac_tags):
                    lac_tags[index] = lac_tags[index][:-1] + 'B'
