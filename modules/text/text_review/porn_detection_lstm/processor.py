# -*- coding: utf-8 -*-
import io
import numpy as np


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            vocab[line.rstrip()] = int(i)
    return vocab


def get_predict_label(pos_prob):
    """
    Convert the prediction probabilities to label
    """
    # threshold should be (1, 0.5)
    neu_threshold = 0.5
    if pos_prob >= neu_threshold:
        label, key = 1, "porn"
    else:
        label, key = 0, "not_porn"
    return label, key


def preprocess(predicted_data, tokenizer, vocab, sequence_max_len=256):
    """
    Convert the word str to word id and pad the text
    """
    result = []
    padding = vocab['<PAD>']
    unknown = vocab['<UNK>']
    for index, text in enumerate(predicted_data):
        data_arr = tokenizer.tokenize(''.join(text.split()))
        wids = [vocab.get(w, unknown) for w in data_arr[:sequence_max_len]]
        if len(wids) < sequence_max_len:
            wids = wids + [padding] * (sequence_max_len - len(wids))

        result_i = {'processed': []}
        result_i['origin'] = predicted_data[index]
        result_i['processed'] += wids
        result.append(result_i)
    return result


def postprocess(predict_out, texts):
    """
    Convert model's output tensor to pornography label
    """
    result = []
    predict_out = predict_out.as_ndarray()
    for index in range(len(texts)):
        result_i = {}
        result_i['text'] = texts[index]['origin']
        label = int(np.argmax(predict_out[index]))
        if label == 0:
            key = 'not_porn'
        else:
            key = 'porn'
        result_i['porn_detection_label'] = label
        result_i['porn_detection_key'] = key
        result_i['porn_probs'] = float('%.4f' % predict_out[index, 1])
        result_i['not_porn_probs'] = float('%.4f' % (predict_out[index, 0]))
        result.append(result_i)
    return result
