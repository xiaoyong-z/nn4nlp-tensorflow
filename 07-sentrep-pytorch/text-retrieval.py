from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import torch
from torch.autograd import Variable
import numpy as np

# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            sent_src = [w2i_src[x] for x in line_src.strip().split()]
            sent_trg = [w2i_trg[x] for x in line_trg.strip().split()]
            yield (sent_src, sent_trg)

# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))

# Model parameters
EMB_SIZE = 64
HID_SIZE = 64
BATCH_SIZE = 16



class bilstm(torch.nn.Module):
    def __init__(self, nwords_src, nwords_trg, EMB_SIZE, HID_SIZE, use_cuda):
        super(bilstm, self).__init__()
        self.useCuda = use_cuda
        self.hidSize = HID_SIZE
        self.embeddingSRC = torch.nn.Embedding(nwords_src, EMB_SIZE)
        self.embeddingTRG = torch.nn.Embedding(nwords_trg, EMB_SIZE)
        torch.nn.init.uniform_(self.embeddingSRC.weight, -0.25, 0.25)
        torch.nn.init.uniform_(self.embeddingTRG.weight, -0.25, 0.25)
        self.srcLstm = torch.nn.LSTM(input_size=EMB_SIZE, hidden_size=HID_SIZE, num_layers=1, bidirectional=True, batch_first=True)
        self.trgLstm = torch.nn.LSTM(input_size=EMB_SIZE, hidden_size=HID_SIZE, num_layers=1, bidirectional=True, batch_first=True)


    def forward(self, sent, isSrc):
        if isSrc:
            sentEmb = self.embeddingSRC(sent)
            if use_cuda:
                srch0 = Variable(torch.zeros(2, 1, self.hidSize).cuda())
                srcc0 = Variable(torch.zeros(2, 1, self.hidSize).cuda())
            else:
                srch0 = Variable(torch.zeros(2, 1, self.hidSize))
                srcc0 = Variable(torch.zeros(2, 1, self.hidSize))
            self.srchidden = (srch0, srcc0)
            sentEmb = sentEmb.unsqueeze(0)

            output, _ = self.srcLstm(sentEmb, self.srchidden)
        else:
            sentEmb = self.embeddingTRG(sent)
            if use_cuda:
                trgh0 = Variable(torch.zeros(2, 1, self.hidSize).cuda())
                trgc0 = Variable(torch.zeros(2, 1, self.hidSize).cuda())
            else:
                trgh0 = Variable(torch.zeros(2, 1, self.hidSize))
                trgc0 = Variable(torch.zeros(2, 1, self.hidSize))
            self.trghidden = (trgh0, trgc0)
            sentEmb = sentEmb.unsqueeze(0)

            output, _ = self.trgLstm(sentEmb, self.trghidden)
        return output[:, -1,:]


type = torch.LongTensor
use_cuda = torch.cuda.is_available()

model = bilstm(nwords_src, nwords_trg, EMB_SIZE, HID_SIZE, use_cuda)
criterion = torch.nn.MultiMarginLoss(reduce=False)
optimizer = torch.optim.Adam(model.parameters())

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()

def calc_loss(sents):
    srcSent = [torch.tensor(src).type(type) for src, tag in sents]
    trgSent = [torch.tensor(tag).type(type) for src, tag in sents]


    src_reps = [model(sent, True) for sent in srcSent]
    trg_reps = [model(sent, False) for sent in trgSent]

    src_mtx = torch.cat(src_reps)
    trg_mtx = torch.cat(trg_reps)


    sim_mtx = torch.matmul(src_mtx, trg_mtx.transpose(1, 0))
    y = torch.tensor(list(range(len(sents)))).type(type)
    loss = criterion(input=sim_mtx, target=y)

    return torch.sum(loss)


# Calculate representations for one corpus
def index_corpus(sents):
    # To take advantage of auto-batching, do several at a time
    for sid in range(0, len(sents), BATCH_SIZE):
        srcSent = [torch.tensor(src).type(type) for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]
        trgSent = [torch.tensor(tag).type(type) for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]

        src_exprs = [model(sent, True) for sent in srcSent]
        trg_exprs = [model(sent, False) for sent in trgSent]

        for src_expr, trg_expr in zip(src_exprs, trg_exprs):
            yield (src_expr.data.numpy()[0], trg_expr.data.numpy()[0])


# Perform retrieval, and return both scores and ranked order of candidates
def retrieve(src, db_mtx):
    scores = np.dot(db_mtx, src)
    ranks = np.argsort(-scores)
    return ranks, scores


# Perform training
start = time.time()
train_mbs = all_time = dev_time = all_tagged = this_sents = this_loss = 0
for ITER in range(100):
    random.shuffle(train)
    for sid in range(0, len(train), BATCH_SIZE):
        my_size = min(BATCH_SIZE, len(train)-sid)
        train_mbs += 1
        if train_mbs % int(1000/BATCH_SIZE) == 0:
            print("loss/sent=%.4f, sent/sec=%.4f" % (this_loss / this_sents, (train_mbs * BATCH_SIZE) / (time.time() - start - dev_time)), file=sys.stderr)
            this_loss = this_sents = 0
        # train on the minibatch
        loss_exp = calc_loss(train[sid:sid+BATCH_SIZE])
        this_loss += loss_exp.item()
        this_sents += BATCH_SIZE
        optimizer.zero_grad()
        loss_exp.backward()
        optimizer.step()
    # Perform evaluation
    dev_start = time.time()
    rec_at_1, rec_at_5, rec_at_10 = 0, 0, 0
    reps = list(index_corpus(dev))
    trg_mtx = np.stack([trg for src, trg in reps])
    for i, (src, trg) in enumerate(reps):
        ranks, scores = retrieve(src, trg_mtx)
        if ranks[0] == i: rec_at_1 += 1
        if i in ranks[:5]: rec_at_5 += 1
        if i in ranks[:10]: rec_at_10 += 1
    dev_time += time.time()-dev_start
    print("epoch %r: dev recall@1=%.2f%% recall@5=%.2f%% recall@10=%.2f%%" % (ITER, rec_at_1/len(dev)*100, rec_at_5/len(dev)*100, rec_at_10/len(dev)*100))
