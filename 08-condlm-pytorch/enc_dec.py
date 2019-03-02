from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import torch
import numpy as np
from model import Seq2Seq

# much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"
test_src_file = "../data/parallel/test.ja"
test_trg_file = "../data/parallel/test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            # need to append EOS tags to at least the target sentence
            sent_src = [w2i_src[x] for x in line_src.strip().split() + ['</s>']]
            sent_trg = [w2i_trg[x] for x in ['<s>'] + line_trg.strip().split() + ['</s>']]
            yield (sent_src, sent_trg)


# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
eos_src = w2i_src['</s>']
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
eos_trg = w2i_trg['</s>']
sos_trg = w2i_trg['<s>']
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
i2w_trg = {v: k for k, v in w2i_trg.items()}

nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))
test = list(read(test_src_file, test_trg_file))

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16

MAX_SENT_SIZE = 50
num_layer = 1

use_cuda = torch.cuda.is_available()
type = torch.LongTensor
model = Seq2Seq(nwords_src, nwords_trg, 1, EMBED_SIZE, HIDDEN_SIZE, MAX_SENT_SIZE, sos_trg, eos_trg, use_cuda)

optimizer = torch.optim.Adam(model.parameters())

if use_cuda:
    type = torch.cuda.is_available()
    model.cuda()


def calc_loss(sent):
    # Transduce all batch elements with an LSTM
    src = sent[0]
    trg = sent[1]
    src_tensor = torch.tensor(src).type(type)
    trg_tensor = torch.tensor(trg).type(type)
    loss, _ = model(src_tensor, trg_tensor)
    loss = torch.stack(loss).sum()
    return loss


def generate(sent):
    src = sent
    src_tensor = torch.tensor(src).type(type)
    _, trg_sent = model(src_tensor)
    for i in range(len(trg_sent)):
        trg_sent[i] = i2w_trg[trg_sent[i]]
    return trg_sent


for ITER in range(1):
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_loss(sent)
        train_loss += my_loss.item()
        train_words += len(sent)
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        if (sent_id + 1) % 1000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
            break
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_loss(sent)
        dev_loss += my_loss.item()
        dev_words += len(sent)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start))

# this is how you generate, can replace with desired sentenced to generate
sentences = []
for sent_id, sent in enumerate(test):
    translated_sent = generate(sent[0])
    sentences.append(translated_sent)
for sent in sentences:
    print(sent)
