from __future__ import print_function
import torch.nn as nn
from torch.autograd import Variable
import torch as t
import torch.cuda as torch
import torch.nn.functional as F
import time

from collections import defaultdict
import random
import math
import sys
import argparse
import numpy as np

# much of the beginning is the same as the text retrieval
# format of files: each line is "word1 word2 ..." aligned line-by-line
criterion = nn.CrossEntropyLoss()
use_cuda = torch.is_available()
atype = torch.LongTensor


class encoder(nn.Module):
    def __init__(self, nwords_trg, emb_size, hidden_dim, num_layer, use_cuda):
        super(encoder, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.embed = nn.Embedding(nwords_trg, emb_size)
        nn.init.uniform_(self.embed.weight, -0.25, 0.25)
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_dim, num_layers=num_layer,
                            batch_first=True)

    def forward(self, source, src_length=None, hidden=None):
        src_emb = self.embed(source)
        src_emb = src_emb.unsqueeze(0)
        if hidden is None:
            h0 = Variable(t.zeros(self.num_layer, 1, self.hidden_dim).cuda())
            c0 = Variable(t.zeros(self.num_layer, 1, self.hidden_dim).cuda())
            hidden = (h0, c0)

        # if src_length is not None:
        #     src_emb = torch.nn.utils.rnn.pack_padded_sequence(src_emb, src_length, batch_first=True)

        output, enc_h_t = self.lstm(src_emb, hidden)

        # if src_length is not None:
        #     output, _ = torch.nn.utils.rnn.pad_packed_sequence(enc_h, batch_first=True)

        return output[:, -1, :]


class decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, trg, hidden):
        '''
        enc_h  : B x S x H
        prev_s : B x H
        '''
        src_emb = self.embed(trg)
        src_emb = src_emb.unsqueeze(0).unsqueeze(1)
        output, hidden = self.lstm(src_emb, hidden)
        output = self.dec2word(output).squeeze(1)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, src_nword, trg_nword, num_layer, embed_dim, hidden_dim, max_len, trg_soi, trg_eos, use_cuda):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = hidden_dim
        self.trg_nword = trg_nword
        self.max = max_len
        self.trg_soi = trg_soi
        self.trg_eos = trg_eos

        self.encoder = encoder(src_nword, embed_dim, hidden_dim, num_layer, use_cuda)
        self.decoder = decoder(trg_nword, embed_dim, hidden_dim)

    def forward(self, source, target=None):
        total_loss = []
        trg_sent = []
        if target is not None:
            enc_h = self.encoder(source)
            enc_h = enc_h.unsqueeze(1)
            hidden = (enc_h, F.tanh(enc_h))

            for i in range(target.shape[0]):
                output, hidden = self.decoder(target[i], hidden)
                loss = criterion(output, target[i].unsqueeze(0))
                total_loss.append(loss)
        else:
            enc_h = self.encoder(source)
            enc_h = enc_h.unsqueeze(1)
            hidden = (enc_h, F.tanh(enc_h))

            trg_word = t.tensor(self.trg_soi).type(atype)
            i = 0
            for i in range(self.max):
                output, hidden = self.decoder(trg_word, hidden)
                trg_word = torch.argmax(output, dim=1)[0]
                trg_sent.append(trg_word.item())
                if (trg_word == self.trg_eos):
                    break
        return total_loss, trg_sent


train_src_file = "train.ja"
train_trg_file = "train.en"
dev_src_file = "dev.ja"
dev_trg_file = "dev.en"
test_src_file = "test.ja"
test_trg_file = "test.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))


def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r", encoding="utf-8") as f_src, open(fname_trg, "r", encoding="utf-8") as f_trg:
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

model = Seq2Seq(nwords_src, nwords_trg, 1, EMBED_SIZE, HIDDEN_SIZE, MAX_SENT_SIZE, sos_trg, eos_trg, use_cuda)
model.cuda()
optimizer = t.optim.Adam(model.parameters())


def calc_loss(sent):
    # Transduce all batch elements with an LSTM
    src = sent[0]
    trg = sent[1]
    src_tensor = t.tensor(src).type(atype)
    trg_tensor = t.tensor(trg).type(atype)
    loss, _ = model(src_tensor, trg_tensor)
    loss = t.stack(loss).sum()
    return loss


def generate(sent):
    src = sent
    src_tensor = t.tensor(src).type(atype)
    _, trg_sent = model(src_tensor)
    for i in range(len(trg_sent)):
        trg_sent[i] = i2w_trg[trg_sent[i]]
    return trg_sent


for ITER in range(10):
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
