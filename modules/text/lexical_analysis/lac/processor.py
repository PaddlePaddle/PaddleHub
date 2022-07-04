# -*- coding:utf-8 -*-
import io
import os

import numpy as np
import six


class Query(object):

    def __init__(self, lac_query):
        self.set_query(lac_query)

    def set_query(self, lac_query):
        """
        self.lac_query_list = ["我/r", "和/c", "妈妈/n", "经常/d", "过去/v", "那儿/r", "散步/v"]
        self.seg_query_list = ["我", "和", "妈妈", "经常", "过去", "那儿", "散步"]
        self.seg_query_str = "我 和 妈妈 经常 过去 那儿 散步"
        self.ori_query_str = "我和妈妈经常过去那儿散步"
        """
        length = len(lac_query['word'])
        if six.PY2:
            self.lac_query_list = [
                lac_query["word"][index].encode("utf8") + "/" + lac_query["tag"][index].encode("utf8")
                for index in range(length)
            ]
        else:
            self.lac_query_list = [lac_query["word"][index] + "/" + lac_query["tag"][index] for index in range(length)]

        self.seg_query_list = []
        for phrase in self.lac_query_list:
            index = phrase.rfind("/")
            word = phrase[0:index]
            self.seg_query_list.append(word)
        self.seg_query_str = " ".join(self.seg_query_list)
        self.ori_query_str = "".join(self.seg_query_list)


class Bound(object):

    def __init__(self, start_index=0, end_index=0, left_bound=0, right_bound=0, left_char_bound=0, right_char_bound=0):
        self.start_index = start_index  # 命中的词的起始位置，char级别
        self.end_index = end_index  # 命中的词的结束位置，char级别
        self.left_bound = left_bound  # 原分词级别的起始位置
        self.right_bound = right_bound  # 原分词级别的结束位置
        self.left_char_bound = left_char_bound  # 原 char 级别的起始位置
        self.right_char_bound = right_char_bound  # 原 char 级别的结束位置


class Interventer(object):

    def __init__(self, ngram_dict_path, user_dict_path):
        self.ngram_dict_path = ngram_dict_path
        self.user_dict_path = user_dict_path
        self.init_pos_types()
        self.load_dict()

    def init_pos_types(self):
        all_pos_types = "n f s t nr ns nt nw nz v vd vn" \
                + " a ad an d m q r p c u xc w PER LOC ORG TIME"
        self.all_pos_types = set([pos_type.lower() for pos_type in all_pos_types.split(" ")])

    def load_dict(self):
        """load unigram dict and user dict"""
        import ahocorasick
        self.total_count = 0.0
        self.ngram_dict = {}
        print("Loading dict...")
        for line in io.open(self.ngram_dict_path, mode="r", encoding="utf-8"):
            if six.PY2:
                word, pos, wordfreq = line.encode("utf-8").strip('\n').split('\t')
            else:
                word, pos, wordfreq = line.strip('\n').split('\t')
            wordfreq = int(wordfreq)
            if pos.lower() not in self.all_pos_types:
                continue
            assert wordfreq > 0, "Word frequency must be postive integer!"
            self.total_count += wordfreq
            self.ngram_dict[word + "/" + pos] = wordfreq
        for key in self.ngram_dict:
            wordfreq = self.ngram_dict[key]
            self.ngram_dict[key] = np.log(wordfreq / self.total_count)
        self.oov_score = np.log(1 / self.total_count)

        self.user_dict = ahocorasick.Automaton()
        for line in io.open(self.user_dict_path, mode="r", encoding="utf-8"):
            if six.PY2:
                word, pos, wordfreq = line.encode("utf-8").strip('\n').split('\t')
            else:
                word, pos, wordfreq = line.strip('\n').split('\t')
            wordfreq = int(wordfreq)
            assert pos in self.all_pos_types, "Invalid POS type"
            assert wordfreq > 0, "Word frequency must be postive integer!"
            self.ngram_dict[word + "/" + pos] = np.log(wordfreq / self.total_count)
            self.user_dict.add_word(word, (word, pos, wordfreq))
        self.user_dict.make_automaton()

    def find_min_bound(self, match_info, query):
        """
        find minimum Bound for match_word
        """
        end_index, (match_word, pos, wordfreq) = match_info
        start_index = end_index - len(match_word) + 1

        bound = Bound(start_index=start_index, end_index=end_index)

        # find left bound
        query_len = 0
        for word_index, word in enumerate(query.seg_query_list):
            query_len += len(word)
            if query_len > start_index:
                bound.left_bound = word_index
                bound.left_char_bound = query_len - len(word)
                break
        # find right bound
        query_len = 0
        for word_index, word in enumerate(query.seg_query_list):
            query_len += len(word)
            if query_len > end_index:
                bound.right_bound = word_index
                bound.right_char_bound = query_len - 1
                break
        return bound

    def calc_lm_score(self, phrase_list):
        """calculate the language model score"""
        lm_score = 0.0
        if len(phrase_list) == 0:
            return 0.0
        for phrase in phrase_list:
            lm_score += self.ngram_dict.get(phrase, self.oov_score)
        return lm_score / len(phrase_list)

    def get_new_phrase_list(self, match_info, bound, query):
        """
        比较用户词典给出的词和原分词结果，根据打分决定是否替换
        """
        new_phrase_list = []
        phrase_left = query.ori_query_str[bound.left_char_bound:bound.start_index]
        phrase_right = query.ori_query_str[bound.end_index + 1:bound.right_char_bound + 1]
        if phrase_left != "":
            phrase_left += "/" + query.lac_query_list[bound.left_bound].split('/')[1]
            new_phrase_list.append(phrase_left)
        new_phrase_list.append(match_info[1][0] + "/" + match_info[1][1])
        if phrase_right != "":
            phrase_right += "/" + query.lac_query_list[bound.right_bound].split('/')[1]
            new_phrase_list.append(phrase_right)

        new_query_list = query.lac_query_list[0: bound.left_bound] + new_phrase_list + \
                query.lac_query_list[bound.right_bound + 1: ]
        new_lm_score = self.calc_lm_score(new_query_list)
        return new_lm_score, new_phrase_list

    def run(self, query):
        """
        step 1, 用AC自动机检测出匹配到的用户词
        step 2, 每个用户词查找最小分词边界，计算每种分词结果的打分，PK
        step 3, 怎么处理冲突？
          3.a. 假设 AC自动机检测到的关键词都是顺序的，那么只需要考虑前后两个的替换词即可
          3.b. 假如前后两个替换词没有位置冲突，那么直接把前一个加到替换列表里
          3.c. 假如前后两个替换词有冲突，比较分数，舍弃一个，更新上一个替换的位置
        step 4, 最终依次执行替换
        """
        last_bound = None
        last_phrase_list = None
        last_lm_score = None
        all_result = []
        old_lm_score = self.calc_lm_score(query.lac_query_list)

        for match_info in self.user_dict.iter(query.ori_query_str):
            #print "matched: \"%s\" in query: \"%s\"" % (match_info[1][0], query.seg_query_str)
            bound = self.find_min_bound(match_info, query)
            new_lm_score, new_phrase_list = self.get_new_phrase_list(match_info, bound, query)

            # 如果打分比原 LAC 结果低，抛弃用户词典里的结果
            if new_lm_score <= old_lm_score:
                #print >> sys.stderr, "skipped %s, old_lm_score: %.5f, " \
                #        "new_lm_score: %.5f" % (" ".join(new_phrase_list), old_lm_score, new_lm_score)
                continue
            # 遇到的第一个匹配到的结果
            if last_bound is None:
                last_bound = bound
                last_phrase_list = new_phrase_list
                last_lm_score = new_lm_score
                continue
            if bound.left_bound > last_bound.right_bound:
                # 位置上没有冲突，则把上次的结果加到最终结果中去
                all_result.append((last_bound, last_phrase_list))
                last_bound = bound
                last_phrase_list = new_phrase_list
                last_lm_score = new_lm_score
            else:
                # 位置上有冲突
                if new_lm_score > last_lm_score:
                    # 若分数高于上次结果，则覆盖；否则丢弃
                    last_bound = bound
                    last_phrase_list = new_phrase_list
                    last_lm_score = new_lm_score

        if last_bound is not None:
            all_result.append((last_bound, last_phrase_list))

        # 合并所有替换的结果
        final_phrase_list = []
        last_index = -1
        for bound, phrase_list in all_result:
            final_phrase_list += query.lac_query_list[last_index + 1:bound.left_bound] + phrase_list
            last_index = bound.right_bound
        final_phrase_list += query.lac_query_list[last_index + 1:]

        final_result = {'word': [], 'tag': []}
        for phrase in final_phrase_list:
            index = phrase.rfind("/")
            word = phrase[0:index]
            tag = phrase[index + 1:]
            final_result['word'].append(word)
            final_result['tag'].append(tag)

        return final_result


def load_kv_dict(dict_path, reverse=False, delimiter="\t", key_func=None, value_func=None):
    """
    Load key-value dict from file
    """
    result_dict = {}
    for line in io.open(dict_path, "r", encoding='utf8'):
        terms = line.strip("\n").split(delimiter)
        if len(terms) != 2:
            continue
        if reverse:
            value, key = terms
        else:
            key, value = terms
        if key in result_dict:
            raise KeyError("key duplicated with [%s]" % (key))
        if key_func:
            key = key_func(key)
        if value_func:
            value = value_func(value)
        result_dict[key] = value
    return result_dict


def word_to_ids(words, word2id_dict, word_replace_dict, oov_id=None):
    """convert word to word index"""
    word_ids = []
    for word in words:
        word = word_replace_dict.get(word, word)
        word_id = word2id_dict.get(word, oov_id)
        word_ids.append(word_id)

    return word_ids


def parse_result(lines, crf_decode, id2label_dict, interventer=None):
    """Convert model's output tensor into string and tags """
    offset_list = crf_decode.lod()[0]
    crf_decode = crf_decode.copy_to_cpu()
    batch_size = len(offset_list) - 1
    batch_out = []
    for sent_index in range(batch_size):
        begin, end = offset_list[sent_index], offset_list[sent_index + 1]
        sent = lines[sent_index]
        tags = [id2label_dict[str(tag_id[0])] for tag_id in crf_decode[begin:end]]

        if interventer:
            interventer.parse_customization(sent, tags)

        sent_out = []
        tags_out = []
        for ind, tag in enumerate(tags):
            # for the first char
            if len(sent_out) == 0 or tag.endswith("B") or tag.endswith("S"):
                sent_out.append(sent[ind])
                tags_out.append(tag[:-2])
                continue
            sent_out[-1] += sent[ind]
            tags_out[-1] = tag[:-2]

        seg_result = {"word": sent_out, "tag": tags_out}
        batch_out.append(seg_result)

    return batch_out


#         sent_out = []
#         tags_out = []
#         parital_word = ""
#         for ind, tag in enumerate(tags):
#             # for the first word
#             if parital_word == "":
#                 parital_word = sent[ind]
#                 tags_out.append(tag.split('-')[0])
#                 continue
#             # for the beginning of word
#             if tag.endswith("-B") or (tag == "O" and tags[ind - 1] != "O"):
#                 sent_out.append(parital_word)
#                 tags_out.append(tag.split('-')[0])
#                 parital_word = sent[ind]
#                 continue
#             parital_word += sent[ind]
#         # append the last word, except for len(tags)=0
#         if len(sent_out) < len(tags_out):
#             sent_out.append(parital_word)
#         seg_result = {"word": sent_out, "tag": tags_out}

#         batch_out.append(seg_result)
#     return batch_out
