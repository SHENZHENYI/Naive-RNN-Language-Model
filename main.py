from torch import nn, optim
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from model import RnnModel
from loader import load_raw_data, get_minibatch, encode_str_data, decode_str_data

def train(n_epochs, model, raw_data, loss_fn, optimizer, device, batchsize=8, seq_len=100, pred_seq_len=200, vocab_size=27):
    '''Train

    This function trains the model by wrapping 'train_one_epoch'

    Args:
        n_epochs: number of epochs to be trained
        model: the model
        raw_data: one big string that holds all training data
        loss_fn: loss function
        optimizer: optimizer

    Returns:
        no returns. the model will be updated.
    '''
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, raw_data[:500000], loss_fn, optimizer, batchsize, device, seq_len, vocab_size)
        print(f'epoch: {epoch}, loss: {loss}')
        predict(model, raw_data, seq_len=pred_seq_len)

def train_one_epoch(model, raw_data, loss_fn, optimizer, batchsize, device, seq_len, vocab_size):
    '''Training process in one epoch

    Args:
        model: the neural network
        raw_data: one big string that holds all training data
        loss_fn: loss function
        optimizer: optimizer
        batchsize: the batch size in the training process
        seq_len: the seq length in the training process

    Returns:
        the training loss after training the epoch.
        model is not returned, but will be updated.
    '''
    model.train()
    n_minibatches = len(raw_data)//(seq_len*batchsize)
    loss_meter = []
    with tqdm(total=(n_minibatches)) as prog:
        for iter_, batch in enumerate(get_minibatch(raw_data, batchsize, seq_len, shuffle=True)):
            hidden = None
            optimizer.zero_grad()
            x = batch[:-1, :].to(device) # F.one_hot(batch[:-1,:], num_classes=95).to(torch.float32)
            y = batch[1:,:].to(device)
            outputs, _ = model(x, hidden)
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.append(loss.item())
    return sum(loss_meter)/len(loss_meter)

def predict(model, raw_data, seq_len=200):
    '''Predict a sequence with a length of 200 when fed with a random char

    Args:
        model: the rnn
        raw_data: the training data, we will only sample one char from it, nothing elses
        seq_len: seq len of the predicted linen

    Returns:
        the result predicted string
    '''
    model.eval()
    hidden = None
    rand_index = 25
    input = encode_str_data(raw_data[rand_index: rand_index + 1])
    #input = F.one_hot(encode_str_data(input_seq), num_classes=95).to(torch.float32)
    out_str = []
    for i in range(seq_len):
        output, hidden = model(input.view(1,-1).to(device), hidden)
        probs = F.softmax(output.view(-1)).cpu().tolist() # to prob
        sample = random.choices(char_set, weights=probs)[0] # sample with the prob
        input = encode_str_data([sample])
        out_str.append(sample)
    print(''.join(out_str))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'my device is {device}')

    char_set = ['n', 'm', 'b', 'w', 'k', 't', 'h', 'z', 'r', 'e', 'l', 'a', 'g', 'i', 'p', 'v', 'o', 'q', 'j', 'f', 'd', 'x', ' ', 's', 'c', 'u', 'y']
    data_path = './data/paul_graham_essay.txt'
    raw_data = load_raw_data(data_path)
    rnn = RnnModel(vocab_size=27, emb_size=27, hidden_size=128, num_layers=2).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    train(100, rnn, raw_data, loss_fn, optimizer, device, batchsize=8, seq_len=25, pred_seq_len=25)
    
if __name__ == '__main__':
    main()