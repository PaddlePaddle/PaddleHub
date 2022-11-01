# -*- coding: utf-8 -*-
import io


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            vocab[parts[0]] = int(parts[1])
    vocab["<unk>"] = len(vocab)
    return vocab


text_a_key = "text_1"
text_b_key = "text_2"


def preprocess(lac, word_dict, data_dict, use_gpu=False, batch_size=1):
    """
    Convert the word str to word id and pad the text
    """
    result = {text_a_key: [], text_b_key: []}
    processed_a = lac.lexical_analysis(data={'text': data_dict[text_a_key]}, use_gpu=use_gpu, batch_size=batch_size)
    processed_b = lac.lexical_analysis(data={'text': data_dict[text_b_key]}, use_gpu=use_gpu)
    unk_id = word_dict['<unk>']
    for index, (text_a, text_b) in enumerate(zip(processed_a, processed_b)):
        result_i = {'processed': []}
        result_i['origin'] = data_dict[text_a_key][index]
        for word in text_a['word']:
            _index = word_dict.get(word, unk_id)
            result_i['processed'].append(_index)
        result[text_a_key].append(result_i)

        result_i = {'processed': []}
        result_i['origin'] = data_dict[text_b_key][index]
        for word in text_b['word']:
            _index = word_dict.get(word, unk_id)
            result_i['processed'].append(_index)
        result[text_b_key].append(result_i)
    return result


def postprocess(pred, data_info):
    """
    Convert model's output tensor to pornography label
    """
    result = []
    for index in range(len(data_info[text_a_key])):
        result_i = {}
        result_i[text_a_key] = data_info[text_a_key][index]['origin']
        result_i[text_b_key] = data_info[text_b_key][index]['origin']
        result_i['similarity'] = float('%.4f' % pred[index][0])
        result.append(result_i)
    return result
