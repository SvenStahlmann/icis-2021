import numpy as np
import pandas as pd


def load_validation_data(valid_in_cat_path, valid_out_of_cat_path):
    valid_in_cat = pd.read_csv(valid_in_cat_path, delimiter=',')
    valid_out_of_cat = pd.read_csv(valid_out_of_cat_path, delimiter=',')

    return valid_in_cat, valid_out_of_cat


def load_fold_data(path, current_fold):
    fold_train = pd.read_csv(path + 'fold-' + str(current_fold) + '-train.csv', delimiter=',')
    fold_test = pd.read_csv(path + 'fold-' + str(current_fold) + '-test.csv', delimiter=',')

    return fold_train, fold_test


def load_all_data_and_labels(base_path, current_fold, test_path, valid_path):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    train, _ = load_fold_data(base_path, current_fold)
    test, valid = load_validation_data(test_path, valid_path)

    x_train = train['text'].tolist()
    x_train = [s.strip() for s in x_train]
    x_test = test['text'].tolist()
    x_test = [s.strip() for s in x_test]
    x_valid = valid['text'].tolist()
    x_valid = [s.strip() for s in x_valid]

    dic_one_hot = {0: [1, 0], 1: [0, 1]}
    y_train = np.array([dic_one_hot.get(n, n) for n in train['labels'].tolist()])
    y_test = np.array([dic_one_hot.get(n, n) for n in test['labels'].tolist()])
    y_valid = np.array([dic_one_hot.get(n, n) for n in valid['labels'].tolist()])

    return x_train, x_test, x_valid, y_train, y_test, y_valid


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors
