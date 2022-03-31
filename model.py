import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class RnnModel(nn.Module):
    '''RNN model

    The architecuture of the RNN model

    the architecture is like: input -> rnn(k layers) -> linear -> output
    '''
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(RnnModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        '''Forward propagation

        Args:
            input: dimention = (seq_len, batchsize, n_feature)
            hidden: dimension = (n_layers, batchsize, out_size), if hidden is None, it will be zeros by default
        Returns:
            out: output
        '''
        emb = self.embedding(input)
        out, hiddens = self.rnn(emb, hidden)
        out = self.linear(out)
        return out, (hiddens[0].detach(), hiddens[1].detach())

if __name__ == '__main__':
    # just for test
    rnn = RnnModel(95, 512, 3)
    x = torch.empty(100, 8, 95)
    out, hid = rnn(x, None)
    print(out.shape, hid[0].shape, hid[1].shape)


