from __future__ import print_function
import time

start = time.time()

from collections import Counter, defaultdict
import random
import math
import sys
import argparse
from torch.autograd import Variable
import torch

# format of files: each line is "word1 word2 ..."
train_file = "../data/ptb/train.txt"
test_file = "../data/ptb/valid.txt"

w2i = defaultdict(lambda: len(w2i))


def read(fname):
    """
    Read a file where each line is of the form "word1 word2 ..."
    Yields lists of the form [word1, word2, ...]
    """
    with open(fname, "r") as fh:
        for line in fh:
            sent = [w2i[x] for x in line.strip().split()]
            sent.append(w2i["<s>"])
            yield sent


train = list(read(train_file))
nwords = len(w2i)
test = list(read(test_file))
S = w2i["<s>"]
assert (nwords == len(w2i))

EMBED_SIZE = 64
HIDDEN_SIZE = 128

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

def convert_to_variable(words):
  var = Variable(torch.LongTensor(words))
  if use_cuda:
    var = var.cuda()

  return var

class lmLstm(torch.nn.Module):
    def __init__(self, nwords, EMBED_SIZE, HIDDEN_SIZE, use_cuda):
        super(lmLstm, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = HIDDEN_SIZE

        self.embedding = torch.nn.Embedding(nwords, EMBED_SIZE)
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.lstm = torch.nn.LSTM(input_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, num_layers=1)
        self.project_layer = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=nwords, bias=True)
        torch.nn.init.xavier_uniform(self.project_layer.weight)

    def forward(self, words):
        embedWord = self.embedding(words)
        if self.use_cuda:
            h0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
            c0 = Variable(torch.zeros(1, 1, self.hidden_size).cuda())
        else:
            h0 = Variable(torch.zeros(1, 1, self.hidden_size))
            c0 = Variable(torch.zeros(1, 1, self.hidden_size))
        self.hidden = (h0, c0)
        embedWord = embedWord.unsqueeze(1)

        output, _ = self.lstm(embedWord, self.hidden)
        output = self.project_layer(output)
        output = torch.squeeze(output, 1)
        loss = torch.nn.functional.cross_entropy(output, convert_to_variable(words), size_average=False)
        return loss





model = lmLstm(nwords, EMBED_SIZE, HIDDEN_SIZE, use_cuda)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()




# Build the language model graph
def calc_lm_loss(sent):

    # get the wids and masks for each step
    totWords = len(sent)
    sent = [S] + sent
    sentTensor = torch.tensor(sent).type(type)
    loss = model(sentTensor)

    return loss, totWords


# Sort training sentences in descending order and count minibatches
train_order = list(range(len(train)))

print("startup time: %r" % (time.time() - start))
# Perform training
start = time.time()
i = all_time = dev_time = all_tagged = this_words = this_loss = 0
for ITER in range(100):
    random.shuffle(train_order)
    for sid in train_order:
        i += 1
        if i % int(500) == 0:
            print(this_loss / this_words, file=sys.stderr)
            all_tagged += this_words
            this_loss = this_words = 0
            all_time = time.time() - start
        if i % int(10000) == 0:
            dev_start = time.time()
            dev_loss = dev_words = 0
            for sent in test:
                loss_exp, mb_words = calc_lm_loss(sent)
                dev_loss += loss_exp.item()
                dev_words += mb_words
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
            dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words, train_time, all_tagged / train_time))
        # train on the minibatch
        loss_exp, mb_words = calc_lm_loss(train[sid])
        this_loss += loss_exp.item()
        this_words += mb_words
        optimizer.zero_grad()
        loss_exp.backward()
        optimizer.step()
    print("epoch %r finished" % ITER)
