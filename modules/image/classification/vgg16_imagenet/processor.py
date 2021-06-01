# coding=utf-8
def load_label_info(file_path):
    with open(file_path, 'r') as fr:
        return fr.read().split("\n")[:-1]
