from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import tensorflow as tf
import numpy as np
from hinge_loss import sparse_multiclass_hinge_loss

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


word_src = tf.placeholder(dtype=tf.int32, shape=[None, None])
word_trg = tf.placeholder(dtype=tf.int32, shape=[None, None])
src_seq_length = tf.placeholder(dtype=tf.int32, shape=[None])
trg_seq_length = tf.placeholder(dtype=tf.int32, shape=[None])
embeddingSRC = tf.get_variable(name="embedding1", shape=(nwords_src, EMB_SIZE), initializer=tf.initializers.random_uniform(0.25, 0.25))
embeddingTRG = tf.get_variable(name="embedding2", shape=(nwords_trg, EMB_SIZE), initializer=tf.initializers.random_uniform(0.25, 0.25))
embededSRC = tf.nn.embedding_lookup(embeddingSRC, word_src)
embededTRG = tf.nn.embedding_lookup(embeddingTRG, word_trg)


lstm_fw_cell_SRC = tf.nn.rnn_cell.LSTMCell(HID_SIZE)
lstm_fw_cell_TRG = tf.nn.rnn_cell.LSTMCell(HID_SIZE)
lstm_bw_cell_SRC = tf.nn.rnn_cell.LSTMCell(HID_SIZE)
lstm_bw_cell_TRG = tf.nn.rnn_cell.LSTMCell(HID_SIZE)

outputs_SRC ,_  = tf.nn.bidirectional_dynamic_rnn(sequence_length=src_seq_length, cell_fw= lstm_fw_cell_SRC,cell_bw= lstm_bw_cell_TRG,inputs= embededSRC,dtype= tf.float32)
outputs_TRG ,_  = tf.nn.bidirectional_dynamic_rnn(sequence_length=trg_seq_length, cell_fw= lstm_fw_cell_SRC,cell_bw= lstm_bw_cell_TRG,inputs= embededTRG,dtype= tf.float32)

src = tf.concat(outputs_SRC, 2)
trg = tf.concat(outputs_TRG, 2)
src_mtx = [src[i][src_seq_length[i] - 1] for i in range(BATCH_SIZE)]
trg_mtx = [trg[i][trg_seq_length[i] - 1] for i in range(BATCH_SIZE)]
src_mtx = tf.stack(src_mtx)
trg_mtx = tf.stack(trg_mtx)
sim_mtx = tf.matmul(src_mtx, tf.transpose(trg_mtx, (1, 0)))

y = tf.constant(list(range(BATCH_SIZE)))
# multi hinge loss is not well implemented in tensorflow, so the performance is not very good. You can try it with hinge_loss.py
# So i replace it with cross entropy here,However, the performance is still a bit worse than the dynet version, recall@!0 can only achieve 38 here.
# And i have no idea how to improve this.
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=sim_mtx)

optimizer = tf.train.AdamOptimizer().minimize(loss)



session = tf.Session()
session.run(tf.global_variables_initializer())


def pad(sents, sentsLength):
    maxLength = np.max(sentsLength)
    padSent = np.zeros((len(sents), maxLength))
    for i in range(len(sents)):
        for j in range(len(sents[i])):
            padSent[i][j] = sents[i][j]
    return padSent

def calc_loss(sents):

    src = [src for src, tag in sents]
    trg = [tag for src, tag in sents]
    srcSentLength = [len(src) for src, tag in sents]
    trgSentLength = [len(tag) for src, tag in sents]
    srcSent = pad(src, srcSentLength)
    trgSent = pad(trg, trgSentLength)


    feed_train = {word_src: srcSent,
                  word_trg: trgSent,
                  src_seq_length: srcSentLength,
                  trg_seq_length: trgSentLength}
    my_loss, _ = session.run([loss, optimizer], feed_dict=feed_train)

    return my_loss


# Calculate representations for one corpus
def index_corpus(sents):
    # To take advantage of auto-batching, do several at a time
    for sid in range(0, len(sents), BATCH_SIZE):
        src = [src for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]
        trg = [tag for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]
        srcSentLength = [len(src) for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]
        trgSentLength = [len(tag) for src, tag in sents[sid:min(sid + BATCH_SIZE, len(sents))]]
        srcSent = pad(src, srcSentLength)
        trgSent = pad(trg, trgSentLength)
        if len(srcSent) == BATCH_SIZE:
            feed_train= {word_src: srcSent,
                          word_trg: trgSent,
                          src_seq_length: srcSentLength,
                          trg_seq_length: trgSentLength}
            src_exprs, trg_exprs = session.run([src_mtx, trg_mtx], feed_dict=feed_train)

            for src_expr, trg_expr in zip(src_exprs, trg_exprs):
                yield (src_expr, trg_expr)


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
        this_loss += np.sum(loss_exp)
        this_sents += BATCH_SIZE
    # Perform evaluation
    dev_start = time.time()
    rec_at_1, rec_at_5, rec_at_10 = 0, 0, 0
    reps = list(index_corpus(dev))
    trg_mtxs = np.stack([trg for src, trg in reps])
    for i, (src, trg) in enumerate(reps):
        ranks, scores = retrieve(src, trg_mtxs)
        if ranks[0] == i: rec_at_1 += 1
        if i in ranks[:5]: rec_at_5 += 1
        if i in ranks[:10]: rec_at_10 += 1
    dev_time += time.time()-dev_start
    print("epoch %r: dev recall@1=%.2f%% recall@5=%.2f%% recall@10=%.2f%%" % (ITER, rec_at_1/len(dev)*100, rec_at_5/len(dev)*100, rec_at_10/len(dev)*100))
