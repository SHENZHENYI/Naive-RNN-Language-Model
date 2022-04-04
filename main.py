from torch import nn, optim
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random

from model import RnnModel
from loader import load_raw_data, get_minibatch, encode_str_data

def train(n_epochs, model, raw_data, loss_fn, optimizer, device, vocab, model_path, batchsize=8, seq_len=100, pred_seq_len=200):
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
        loss = train_one_epoch(model, raw_data, loss_fn, optimizer, batchsize, device, seq_len, vocab, model_path)
        print(f'epoch: {epoch}, loss: {loss}')
        predict(model, raw_data, device, vocab, seq_len=pred_seq_len)

def train_one_epoch(model, raw_data, loss_fn, optimizer, batchsize, device, seq_len, vocab, model_path):
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
        for iter_, batch in enumerate(get_minibatch(raw_data, vocab, batchsize, seq_len, shuffle=True)):
            hidden = None
            optimizer.zero_grad()
            x = batch[:-1, :].to(device) # F.one_hot(batch[:-1,:], num_classes=95).to(torch.float32)
            y = batch[1:,:].to(device)
            outputs, _ = model(x, hidden)
            loss = loss_fn(outputs.view(-1, len(vocab)), y.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            prog.update(1)
            loss_meter.append(loss.item())
            save_model(model, model_path)
    return sum(loss_meter)/len(loss_meter)

def predict(model, raw_data, device, vocab, seed=None, seq_len=200):
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
    rand_index = random.randrange(0, len(raw_data)-seq_len)
    if seed is None:
        seed = raw_data[rand_index]
    input = encode_str_data(seed, vocab)
    #input = F.one_hot(encode_str_data(input_seq), num_classes=95).to(torch.float32)
    out_str = []
    for i in range(seq_len):
        output, hidden = model(input.view(1,-1).to(device), hidden)
        probs = F.softmax(output.view(-1)).cpu().tolist() # to prob
        sample = random.choices(vocab, weights=probs)[0] # sample with the prob
        input = encode_str_data(sample, vocab)
        out_str.append(sample)
    print('*'*60)
    print(f'The predicted string with a starting seed of {seed}')
    print('*'*60)
    print(seed+''.join(out_str))
    print('*'*60)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def load_model(model, save_path, device):
    model.load_state_dict(torch.load(save_path, map_location=device))

def main():
    training_mode = 0 # train or inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'my device is {device}')

    data_path = './data/sherlock.txt'
    model_path = './weights/sherlock_e15.pth'
    raw_data, vocab = load_raw_data(data_path)
    model = RnnModel(vocab_size=len(vocab), emb_size=len(vocab), hidden_size=512, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.CrossEntropyLoss()

    if training_mode:
        train(15, model, raw_data, loss_fn, optimizer, device, vocab, model_path, batchsize=32, seq_len=100, pred_seq_len=200)
    else:
        starting_char = 'S' # the first char of the predicted lines
        load_model(model, model_path, device)
        predict(model, raw_data, device, vocab, seed=starting_char, seq_len=800)
    
if __name__ == '__main__':
    main()