# -*- coding: utf-8 -*-
'''
Evaluation script for CMRC 2018
version: v5 - special
Note:
v5 - special: Evaluate on SQuAD-style CMRC 2018 Datasets
v5: formatted output, add usage description
v4: fixed segmentation issues
'''
from __future__ import print_function
from collections import OrderedDict

import os
import re

import nltk

from paddlehub.common.dir import THIRD_PARTY_HOME
from paddlehub.common.downloader import default_downloader

_PUNKT_URL = "https://paddlehub.bj.bcebos.com/paddlehub-thirdparty/punkt.tar.gz"


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    nltk_path = os.path.join(THIRD_PARTY_HOME, "nltk_data")
    tokenizers_path = os.path.join(nltk_path, "tokenizers")
    punkt_path = os.path.join(tokenizers_path, "punkt")
    if not os.path.exists(punkt_path):
        default_downloader.download_file_and_uncompress(
            url=_PUNKT_URL, save_path=tokenizers_path, print_progress=True)
    nltk.data.path.append(nltk_path)

    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = [
        '-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=', '，', '。', '：',
        '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、', '「', '」', '（',
        '）', '－', '～', '『', '』'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def evaluate(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for instance in ground_truth_file:
        # context_id   = instance['context_id'].strip()
        # context_text = instance['context_text'].strip()
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                query_text = qas['question'].strip()
                answers = [x["text"] for x in qas['answers']]

                if query_id not in prediction_file:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_file[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def get_eval(original_file, prediction_file):
    F1, EM, TOTAL, SKIP = evaluate(original_file, prediction_file)
    AVG = (EM + F1) * 0.5
    output_result = OrderedDict()
    output_result['AVERAGE'] = AVG
    output_result['F1'] = F1
    output_result['EM'] = EM
    output_result['TOTAL'] = TOTAL
    output_result['SKIP'] = SKIP

    return output_result
