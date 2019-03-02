from collections import defaultdict
import time
import random
import torch
import numpy as np
from torch.autograd import Variable

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

EMB_SIZE = 64
HID_SIZE = 80

type = torch.LongTensor
use_cuda = torch.cuda.is_available()


class LSTM(torch.nn.Module):
    def __init__(self, EMB_SIZE, HID_SIZE, ntags, nwords, use_cuda):
        super(LSTM, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = HID_SIZE

        self.word_emb = torch.nn.Embedding(nwords, EMB_SIZE)
        torch.nn.init.uniform_(self.word_emb.weight, -0.25, 0.25)
        self.bilstm = torch.nn.LSTM(input_size=EMB_SIZE, hidden_size=HID_SIZE, num_layers=1, bidirectional=True)
        self.project_layer = torch.nn.Linear(in_features=2 * HID_SIZE, out_features=ntags, bias=True)
        torch.nn.init.xavier_uniform_(self.project_layer.weight)




    def forward(self, words, batch_size=1):
        embed_word = self.word_emb(words)
        if self.use_cuda:
            h0 = Variable(torch.zeros(2, 1, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(2, 1, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(2, 1, self.hidden_size))
            c0 = Variable(torch.zeros(2, 1, self.hidden_size))
        self.hidden = (h0, c0)
        embed_word = embed_word.unsqueeze(1)

        output, _ = self.bilstm(embed_word, self.hidden)
        output = self.project_layer(output[-1])
        return output


model = LSTM(EMB_SIZE, HID_SIZE, ntags, nwords, use_cuda)
criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.Adam(model.parameters())

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    count = 0
    model.train()
    for words, tag in train:
        word_tensor = torch.tensor(words).type(type)
        tag_tensor = torch.tensor([tag]).type(type)
        my_loss = criterion(model(word_tensor), tag_tensor)
        train_loss += my_loss.item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        train_loss += my_loss
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
    # Perform training
    model.eval()
    test_correct = 0.0
    for words, tag in dev:
        # Padding (can be done in the conv layer as well)
        words_tensor = torch.tensor(words).type(type)
        scores = model(words_tensor)[0]
        predict = scores.argmax().item()
        if predict== tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))