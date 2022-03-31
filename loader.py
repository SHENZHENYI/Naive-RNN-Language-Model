import torch
import numpy as np

def load_raw_data(data_path):
    '''Load raw text data into one big String

    Args:
        data_path: the path of the corpus text file

    Returns:
        string that represents the whole data
    '''
    valid_char_set = set('qwertyuiopasdfghjklzxxcvbnm ')
    invalid_char_set = set(['t', "'", '3', 'm', 'f', 'i', '>', '}', 'p', '2', '-', 'a', 'k', '`', ';', ':', '4', 's', '1', '{', 'v', 'g', 'h', '(', '<', '%', 'x', '&', '"', 'q', ',', 'j', ']', ' ', '#', 'u', 'l', '?', '_', 'w', 'n', 'z', '9', '$', '~', '*', 'y', '|', '0', 'd', '=', 'b', ')', '!', 'é', '@', '8', 'o', '[', '^', 'r', '+', '.', '7', '6', 'e', '/', 'c', '5']) - valid_char_set
    with open(data_path, 'r') as f:
        data = f.read().replace('\n', ' ').lower()
    for char_ in invalid_char_set:
        data = data.replace(char_, '')
    return data

def encode_str_data(data):
    '''Get the data from string to integer

    This function gets a string in, and returns the same context in integer representation encoded by 'char_set'

    Args:
        data: string data

    Returns:
        same data in torch long
    '''
    char_set = ['n', 'm', 'b', 'w', 'k', 't', 'h', 'z', 'r', 'e', 'l', 'a', 'g', 'i', 'p', 'v', 'o', 'q', 'j', 'f', 'd', 'x', ' ', 's', 'c', 'u', 'y']

    char2int = {c: i for i, c in enumerate(char_set)}
    int_data = [char2int[c] for c in data]
    return torch.tensor(int_data).long()

def decode_str_data(data):
    char_set = ['n', 'm', 'b', 'w', 'k', 't', 'h', 'z', 'r', 'e', 'l', 'a', 'g', 'i', 'p', 'v', 'o', 'q', 'j', 'f', 'd', 'x', ' ', 's', 'c', 'u', 'y']

    int2char = {i:c for i, c in enumerate(char_set)}
    str_data = [int2char[i] for i in data]
    return str_data

def get_minibatch(raw_data, batchsize, seq_len, shuffle=True):
    '''Get the minibatch

    This function yields minibatches of the raw data, which can be fed into the model after doing one-hot encoding

    Args:
        raw_data: text data in integer format
        batchsize: batch size
        seq_len: the length of each sequence
        shuffle: if True, the minibatches will be in random order

    Returns:
        yield minibatches of the input data. The dimension is (seq_len, batchsize,)
    '''
    len_data = len(raw_data)
    indices = np.arange(0, len_data, batchsize*seq_len)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in indices:
        if minibatch_start+batchsize*seq_len < len_data:
            yield encode_str_data(raw_data[minibatch_start: minibatch_start+batchsize*seq_len]).view(seq_len, batchsize,)

if __name__ == '__main__':
    data_path = '../corpus/paul_graham_essay.txt'
    data = load_raw_data(data_path).lower()
    print((list(set(data))))


