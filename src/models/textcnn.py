from collections import defaultdict
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

def load_df(fpath):
    df = pd.read_csv(fpath, delimiter=';')
    df['sentece'] = df['sentece'].str.split()
    df = df[["sentece", "need"]]
    print(df.head())

    df = df[df.need.apply(lambda x: (str)(x).isnumeric())]
    print(df.head())
    df['need'] = df['need'].astype(int)

    return df

def createData(df):
    # transform df to data format for pytorch
    df = df[["sentece", "need"]]
    df['sentece'] = df['sentece'].str.split()
    max_sentence_len = (df['sentece'].str.len()).max()
    records = df.to_records(index=False)
    data = list(records)
    print(data[:5])
    return data, max_sentence_len

def createVocab(data):
    # create the vocab
    vocab = []
    for d, _ in data:
        for w in d:
            if w not in vocab: vocab.append(w)
    vocab = sorted(vocab)
    vocab_size = len(vocab)
    print('vocab examples:', vocab[:10])
    print('vocab size', len(vocab))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    return w2i,i2w , vocab_size


class Net(nn.Module):
    def __init__(self, vocab_size, embd_size, out_chs, filter_heights):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        self.conv = nn.ModuleList([nn.Conv2d(1, out_chs, (fh, embd_size)) for fh in filter_heights])
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(out_chs * len(filter_heights), 1)

    def forward(self, x):
        x = self.embedding(x)  # (N, seq_len, embd_dim)
        x = x.unsqueeze(1)  # (N, Cin, W, embd_dim), insert Channnel-In dim
        # Conv2d
        #    Input : (N,Cin, Hin, Win )
        #    Output: (N,Cout,Hout,Wout)
        # squeeze(3) means 2D to 1D; (N,Cout,Hout,Wout) -> [(N,Cout,Hout==seq_len)] * len(filter_heights)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        # max_pool1d(input, kernel_size, ..
        # (N, Cout, seq_len) --(max_pool1d)--> (N, Cout, 1) --(squeeze(2))--> (N, Cout)
        # [(N, Cout)]  len(filter_heights)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # (N, Cout*len(filter_heights))
        x = self.dropout(x)
        x = self.fc1(x)
        #probs = F.sigmoid(x)
        probs = torch.sigmoid(x)
        return probs


def train(model, data, batch_size, n_epoch):
    model.train()  # Sets the module in training mode. This has any effect only on modules such as Dropout or BatchNorm.
    if use_cuda:
        model.cuda()
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(n_epoch):
        epoch_loss = 0.0
        random.shuffle(data)
        for i in range(0, len(data) - batch_size, batch_size):  # discard some last elements
            in_data, labels = [], []
            for sentence, label in data[i: i + batch_size]:
                index_vec = [w2i[w] for w in sentence]
                pad_len = max(0, max_sentence_len - len(index_vec))
                index_vec += [0] * pad_len
                index_vec = index_vec[:max_sentence_len]  ## TBD for same len
                in_data.append(index_vec)
                labels.append(label)
            sent_var = Variable(torch.LongTensor(in_data))
            if use_cuda: sent_var = sent_var.cuda()

            target_var = Variable(torch.Tensor(labels).unsqueeze(1))
            if use_cuda: target_var = target_var.cuda()
            optimizer.zero_grad()
            probs = model(sent_var)
            loss = F.binary_cross_entropy(probs, target_var)
            loss.backward()
            optimizer.step()

            #epoch_loss += loss.data[0]
            epoch_loss += loss.item()

        print('epoch: {:d}, loss: {:.3f}'.format(epoch, epoch_loss))
        losses.append(epoch_loss)
    print('Training avg loss: {:.3f}'.format(sum(losses) / len(losses)))

    return model, losses


def test(model, data, n_test, min_sentence_len):
    model.eval()
    loss = 0
    correct = 0
    for sentence, label in data[:n_test]:
        if len(sentence) < min_sentence_len:  # to short for CNN's filter
            continue
        index_vec = [w2i[w] for w in sentence]
        sent_var = Variable(torch.LongTensor([index_vec]))
        if use_cuda: sent_var = sent_var.cuda()
        out = model(sent_var)
        score = out.data[0][0]
        pred = 1 if score > .5 else 0
        if pred == label:
            correct += 1
        loss += math.pow((label - score), 2)
    print('Test acc: {:.3f} ({:d}/{:d})'.format(correct / n_test, correct, n_test))
    print('Test loss: {:.3f}'.format(loss / n_test))

def train2(data, max_sentence_len, w2i, vocab_size, batch_size = 64, n_epoch = 50, use_cuda = True):
    out_ch = 100
    embd_size = 128
    filter = [1,2,3]
    model = Net(vocab_size, embd_size, out_ch, filter)

    model.train()  # Sets the module in training mode. This has any effect only on modules such as Dropout or BatchNorm.
    if use_cuda:
        model.cuda()
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(n_epoch):
        epoch_loss = 0.0
        random.shuffle(data)
        for i in range(0, len(data) - batch_size, batch_size):  # discard some last elements
            in_data, labels = [], []
            for sentence, label in data[i: i + batch_size]:
                index_vec = [w2i[w] for w in sentence]
                pad_len = max(0, max_sentence_len - len(index_vec))
                index_vec += [0] * pad_len
                index_vec = index_vec[:max_sentence_len]  ## TBD for same len
                in_data.append(index_vec)
                labels.append(label)
            sent_var = Variable(torch.LongTensor(in_data))
            if use_cuda: sent_var = sent_var.cuda()

            target_var = Variable(torch.Tensor(labels).unsqueeze(1))
            if use_cuda: target_var = target_var.cuda()
            optimizer.zero_grad()
            probs = model(sent_var)
            loss = F.binary_cross_entropy(probs, target_var)
            loss.backward()
            optimizer.step()

            # epoch_loss += loss.data[0]
            epoch_loss += loss.item()

        print('epoch: {:d}, loss: {:.3f}'.format(epoch, epoch_loss))
        losses.append(epoch_loss)
    print('Training avg loss: {:.3f}'.format(sum(losses) / len(losses)))
    return model

def predict(model, data, w2i, min_sentence_len, use_cuda = True):
    model.eval()
    loss = 0
    correct = 0
    prediction = []
    for sentence, label in data:
        if len(sentence) < min_sentence_len:  # to short for CNN's filter
            print(f"sentence '{sentence}' is to short for the filter, please prediction set to -1")
            prediction.append(0)
            continue
        index_vec = [w2i[w] for w in sentence]
        sent_var = Variable(torch.LongTensor([index_vec]))
        if use_cuda: sent_var = sent_var.cuda()
        out = model(sent_var)
        score = out.data[0][0]
        pred = 1 if score > .5 else 0
        prediction.append(pred)
        if pred == label:
            correct += 1
        loss += math.pow((label - score), 2)
    print('Test acc: {:.3f} ({:d}/{:d})'.format(correct / len(data), correct, len(data)))
    print('Test loss: {:.3f}'.format(loss / len(data)))

    return prediction

if __name__ == '__main__':
    use_cuda = True


    df = load_df('../../data/raw/amazon-reviews4.csv')

    max_sentence_len = (df['sentece'].str.len()).max()

    records = df.to_records(index=False)
    data = list(records)
    print(data[:5])

    print('sentence maxlen', max_sentence_len)

    vocab = []
    for d, _ in data:
        for w in d:
            if w not in vocab: vocab.append(w)
    vocab = sorted(vocab)
    vocab_size = len(vocab)
    print('vocab examples:', vocab[:10])
    print('vocab size', len(vocab))

    w2i = {w:i for i,w in enumerate(vocab)}
    i2w = {i:w for i,w in enumerate(vocab)}

    # split data into train and test data
    div_idx = (int)(len(data) * 0.8)
    random.shuffle(data)
    train_data = data[:div_idx]
    test_data = data[div_idx:]
    print('n_train', len(train_data))
    print('n_test', len(test_data))

    out_ch = 100
    embd_size = 128
    batch_size = 64
    n_epoch = 50
    filter_variations = [[1], [1, 2], [1, 2, 3, 4]]
    for fil in filter_variations:
        print('filter:', fil)
        model = Net(vocab_size, embd_size, out_ch, fil)
        #     print(model)
        model, losses = train(model, train_data, batch_size, n_epoch)
        test(model, test_data, len(test_data), max(fil))
        print('')



