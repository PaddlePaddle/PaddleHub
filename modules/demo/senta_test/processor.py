def load_vocab(vocab_path):
    with open(vocab_path) as file:
        return file.read().split()
