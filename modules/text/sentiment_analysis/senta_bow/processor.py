# -*- coding:utf-8 -*-
import io
import numpy as np


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            parts = line.rstrip().split('\t')
            vocab[parts[0]] = int(parts[1])
    vocab["<unk>"] = len(vocab)
    return vocab


def preprocess(lac, texts, word_dict, use_gpu=False, batch_size=1):
    """
    firstly, the predicted texts are segmented by lac module
    then, the word segmention results input into senta
    """
    result = []
    input_dict = {"text": texts}
    processed = lac.lexical_analysis(data=input_dict, use_gpu=use_gpu, batch_size=batch_size)
    unk_id = word_dict["<unk>"]
    for index, data in enumerate(processed):
        result_i = {'processed': []}
        result_i['origin'] = texts[index]
        for word in data['word']:
            if word in word_dict:
                _index = word_dict[word]
            else:
                _index = unk_id
            result_i['processed'].append(_index)
        result.append(result_i)
    return result


def postprocess(predict_out, texts):
    """
    Convert model's output tensor to sentiment label
    """
    predict_out = predict_out.as_ndarray()
    batch_size = len(texts)
    result = []
    for index in range(batch_size):
        result_i = {}
        result_i['text'] = texts[index]['origin']
        label = int(np.argmax(predict_out[index]))
        if label == 0:
            key = 'negative'
        else:
            key = 'positive'
        result_i['sentiment_label'] = label
        result_i['sentiment_key'] = key
        result_i['positive_probs'] = float('%.4f' % predict_out[index, 1])
        result_i['negative_probs'] = float('%.4f' % (1 - predict_out[index, 1]))
        result.append(result_i)
    return result
