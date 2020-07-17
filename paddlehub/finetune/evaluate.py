#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import math

import numpy as np


# Sequence label evaluation functions
def chunk_eval(np_labels, np_infers, np_lens, tag_num, dev_count=1):
    def extract_bio_chunk(seq):
        chunks = []
        cur_chunk = None
        null_index = tag_num - 1
        for index in range(len(seq)):
            tag = seq[index]
            tag_type = tag // 2
            tag_pos = tag % 2

            if tag == null_index:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = None
                continue

            if tag_pos == 0:
                if cur_chunk is not None:
                    chunks.append(cur_chunk)
                    cur_chunk = {}
                cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

            else:
                if cur_chunk is None:
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}
                    continue

                if cur_chunk["type"] == tag_type:
                    cur_chunk["en"] = index + 1
                else:
                    chunks.append(cur_chunk)
                    cur_chunk = {"st": index, "en": index + 1, "type": tag_type}

        if cur_chunk is not None:
            chunks.append(cur_chunk)
        return chunks

    null_index = tag_num - 1
    num_label = 0
    num_infer = 0
    num_correct = 0
    labels = np_labels.reshape([-1]).astype(np.int32).tolist()
    infers = np_infers.reshape([-1]).astype(np.int32).tolist()
    all_lens = np_lens.reshape([dev_count, -1]).astype(np.int32).tolist()

    base_index = 0
    for dev_index in range(dev_count):
        lens = all_lens[dev_index]
        max_len = 0
        for l in lens:
            max_len = max(max_len, l)

        for i in range(len(lens)):
            seq_st = base_index + i * max_len + 1
            seq_en = seq_st + (lens[i] - 2)
            infer_chunks = extract_bio_chunk(infers[seq_st:seq_en])
            label_chunks = extract_bio_chunk(labels[seq_st:seq_en])
            num_infer += len(infer_chunks)
            num_label += len(label_chunks)

            infer_index = 0
            label_index = 0
            while label_index < len(label_chunks) \
                   and infer_index < len(infer_chunks):
                if infer_chunks[infer_index]["st"] \
                    < label_chunks[label_index]["st"]:
                    infer_index += 1
                elif infer_chunks[infer_index]["st"] \
                    > label_chunks[label_index]["st"]:
                    label_index += 1
                else:
                    if infer_chunks[infer_index]["en"] \
                        == label_chunks[label_index]["en"] \
                        and infer_chunks[infer_index]["type"] \
                        == label_chunks[label_index]["type"]:
                        num_correct += 1

                    infer_index += 1
                    label_index += 1

        base_index += max_len * len(lens)

    return num_label, num_infer, num_correct


def calculate_f1(num_label, num_infer, num_correct):
    if num_infer == 0:
        precision = 0.0
    else:
        precision = num_correct * 1.0 / num_infer

    if num_label == 0:
        recall = 0.0
    else:
        recall = num_correct * 1.0 / num_label

    if num_correct == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def calculate_f1_np(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))
    p = tp / (tp + fp) if (tp + fp) else 0
    r = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * p * r) / (p + r) if p + r else 0
    return p, r, f1


def matthews_corrcoef(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    tp = np.sum((labels == 1) & (preds == 1))
    tn = np.sum((labels == 0) & (preds == 0))
    fp = np.sum((labels == 0) & (preds == 1))
    fn = np.sum((labels == 1) & (preds == 0))

    div = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(div) if div else 0
    return mcc


def recall_nk(data, n, k, m):
    """
    This metric can be used to evaluate whether the model can find the correct response B for question A
    Note: Only applies to each question A only has one correct response B1.

    Parameters
    ----------
    data: List. Each element is a tuple, consist of the positive probability of the sample prediction and its label.
                For each example, the only one true positive sample should be the first tuple.
    n: int. The number of labels per example.
        eg: [A,B1,1], [A,B2,0], [A,B3,0]  n=3 as there has 3 labels for example A
    k: int. If the top k is right, the example will be considered right.
        eg: [A,B1,1]=0.5, [A,B2,0]=0.8, [A,B3,0]=0.3(Probability of 1)
           If k=2, the prediction for the example A will be considered correct as 0.5 is the top2 Probability
           If k=1, the prediction will be considered wrong as 0.5 is not the biggest probability.
    m: int. For every m examples, there's going to be a positive sample.
        eg. data= [A1,B1,1], [A1,B2,0], [A1,B3,0], [A2,B1,1], [A2,B2,0], [A2,B3,0]
           For every 3 examples, there will be one positive sample. so m=3, and n can be 1,2 or 3.
    """

    def get_p_at_n_in_m(data, n, k, ind):
        """
        calculate precision in recall n
        """
        pos_score = data[ind][0]
        curr = data[ind:ind + n]
        curr = sorted(curr, key=lambda x: x[0], reverse=True)
        if curr[k - 1][0] <= pos_score:
            return 1
        return 0

    correct_num = 0.0

    length = len(data) // m

    for i in range(0, length):
        ind = i * m
        assert data[ind][1] == 1

        correct_num += get_p_at_n_in_m(data, n, k, ind)

    return correct_num / length


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def _get_ngrams(segment, max_order):
    """
    Extracts all n-grams upto a given maximum order from an input segment.

    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.

    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 smooth=False):
    """
    Computes BLEU score of translated segments against one or more references.

    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.

    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (reference, translation) in zip(reference_corpus, translation_corpus):
        reference_length += len(reference)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (
                    float(matches_by_order[i]) / possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    elif ratio > 0.0:
        bp = math.exp(1 - 1. / ratio)
    else:
        bp = 0

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)
