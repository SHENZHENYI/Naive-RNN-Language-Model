import torch
import numpy as np
import random

def load_raw_data(data_path):
    '''Load raw text data into one big String, and get the vocabulary of the data at char level

    Args:
        data_path: the path of the corpus text file

    Returns:
        data: string that represents the whole data
        vocab: a vocabulary set of the raw data
    '''
    with open(data_path, 'r') as f:
        data = f.read()
    vocab = sorted(list(set(data)))
    return data, vocab

def encode_str_data(data, vocab):
    '''Get the data from string to integer

    This function gets a string in, and returns the same context in integer representation encoded by 'char_set'

    Args:
        data: list of list of string or just string
        vocab: vocabulary set

    Returns:
        encoded data
    '''
    char2int = {c: i for i, c in enumerate(vocab)}
    if isinstance(data, list):
        int_data = []
        for i in range(len(data)):
            int_data.append([])
            for j in range(len(data[0])):
                int_data[i].append(char2int[data[i][j]])
    elif isinstance(data, str):
        int_data = char2int[data]
    else:
        raise TypeError(f'{data} has a wrong data type')
    return torch.tensor(int_data).long()

def get_minibatch(raw_data, vocab, batchsize, seq_len, shuffle=True):
    '''Get the minibatch

    This function yields minibatches of the raw data, which can be fed into the model directly

    Args:
        raw_data: text data in integer format
        batchsize: batch size
        seq_len: the length of each sequence
        shuffle: if True, the minibatches will be in random order

    Returns:
        yield minibatches of the input data. The dimension is (seq_len, batchsize,)
    '''
    len_data = len(raw_data)
    start_seed = random.randrange(0,200)
    indices = np.arange(start_seed, len_data-seq_len, batchsize)
    if shuffle:
        np.random.shuffle(indices)
    for i in range(len(indices)):
        starts = indices[i:i+batchsize]
        texts = [raw_data[start:start+seq_len] for start in starts]
        yield encode_str_data(texts, vocab).T

if __name__ == '__main__':
    data_path = './data/paul_graham_essay.txt'
    data, vocab = load_raw_data(data_path)
    print(vocab)


