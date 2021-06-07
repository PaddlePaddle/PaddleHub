# -*- coding:utf-8 -*-
import io
import numpy as np


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as fin:
        wid = 0
        for line in fin:
            data = line.strip().split("\t")
            if len(data) == 1:
                wstr = ''
                vocab[wstr] = int(data[0])
                continue
            else:
                wstr = data[0]
            vocab[wstr] = int(data[1])
    vocab["<unk>"] = len(vocab)
    return vocab


def get_predict_label(probs):
    label = int(np.argmax(probs))
    if label == 0:
        key = "negative"
    elif label == 2:
        key = "positive"
    else:
        key = "neutral"
    return label, key


def preprocess(lac, predicted_data, word_dict, use_gpu=False, batch_size=1):
    result = []
    data_dict = {"text": predicted_data}
    processed = lac.lexical_analysis(data=data_dict, use_gpu=use_gpu, batch_size=batch_size)
    unk_id = word_dict["<unk>"]
    for index, data in enumerate(processed):
        result_i = {'processed': []}
        result_i['origin'] = predicted_data[index]
        for word in data['word']:
            if word in word_dict:
                _index = word_dict[word]
            else:
                _index = unk_id
            result_i['processed'].append(_index)
        result.append(result_i)
    return result


def postprocess(prediction, texts):
    result = []
    pred = prediction.as_ndarray()
    for index in range(len(texts)):
        result_i = {}
        result_i['text'] = texts[index]['origin']
        label, key = get_predict_label(pred[index])
        result_i['emotion_label'] = label
        result_i['emotion_key'] = key
        result_i['positive_probs'] = float('%.4f' % pred[index, 2])
        result_i['negative_probs'] = float('%.4f' % (pred[index, 0]))
        result_i['neutral_probs'] = float('%.4f' % (pred[index, 1]))
        result.append(result_i)
    return result
