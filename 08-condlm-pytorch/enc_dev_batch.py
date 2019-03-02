from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import torch
import numpy as np
from model_batch import Seq2Seq

# some of this code borrowed from Qinlan Shen's attention from the MT class last year
# much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


# Creates batches where all source sentences are the same length
def create_batches(sorted_dataset, max_batch_size):
    source = [x[0] for x in sorted_dataset]
    src_lengths = [len(x) for x in source]
    batches = []
    prev = src_lengths[0]
    prev_start = 0
    batch_size = 1
    for i in range(1, len(src_lengths)):
        if src_lengths[i] != prev or batch_size == max_batch_size:
            batches.append((prev_start, batch_size))
            prev = src_lengths[i]
            prev_start = i
            batch_size = 1
        else:
            batch_size += 1
    return batches


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


def calc_loss(sents):
    # Transduce all batch elements with an LSTM
    src_sents = [x[0] for x in sents]
    tgt_sents = [x[1] for x in sents]
    src_cws = []

    src_len = [len(sent) for sent in src_sents]
    max_src_len = np.max(src_len)
    num_words = 0

    for i in range(max_src_len):
        src_cws.append([sent[i] for sent in src_sents])

    tgt_cws = []
    tgt_len = [len(sent) for sent in tgt_sents]
    max_tgt_len = np.max(tgt_len)
    masks = []

    for i in range(max_tgt_len):
        tgt_cws.append([sent[i] if len(sent) > i else eos_trg for sent in tgt_sents])
        mask = [(1 if len(sent) > i else 0) for sent in tgt_sents]
        masks.append(mask)
        num_words += sum(mask)

    src_tensor = torch.tensor(src_cws).type(type)
    trg_tensor = torch.tensor(tgt_cws).type(type)
    src_tensor = src_tensor.transpose(1, 0)
    trg_tensor = trg_tensor.transpose(1, 0)
    mask_tensor = torch.tensor(masks).type(type)
    mask_tensor = mask_tensor.transpose(1, 0)
    loss, _ = model(src_tensor, trg_tensor, mask_tensor)
    loss = torch.stack(loss).sum()
    return loss, num_words


def generate(sent):
    src = sent
    src_tensor = torch.tensor(src).type(type)
    _, trg_sent = model(src_tensor)
    for i in range(len(trg_sent)):
        trg_sent[i] = i2w_trg[trg_sent[i]]
    return trg_sent



for ITER in range(100):
    # Perform training
    train.sort(key=lambda t: len(t[0]), reverse=True)
    dev.sort(key=lambda t: len(t[0]), reverse=True)
    train_order = create_batches(train, BATCH_SIZE)
    dev_order = create_batches(dev, BATCH_SIZE)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, (start, length) in enumerate(train_order):
        train_batch = train[start:start + length]
        my_loss, num_words = calc_loss(train_batch)
        train_loss += my_loss.item()
        train_words += num_words
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
    print(generate(dev[0][0]))
    for sent_id, (start, length) in enumerate(dev_order):
        dev_batch = dev[start:start + length]
        my_loss, num_words = calc_loss(dev_batch)
        dev_loss += my_loss.item()
        dev_words += num_words
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start))
