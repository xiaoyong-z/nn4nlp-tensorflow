from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import tensorflow as tf
import numpy as np
import pdb

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
EMB_SIZE = 64
HIDDEN_SIZE = 128

# Especially in early training, the model can generate basically infinitly without generating an EOS
# have a max sent size that you end at
MAX_SENT_SIZE = 50

encoder_src = tf.placeholder(dtype=tf.int32, shape=[None])
decoder_src = tf.placeholder(dtype=tf.int32, shape=[None])
decoder_target = tf.placeholder(dtype=tf.int32, shape=[None])

embeddingSRC = tf.get_variable(name="embedding_src", shape=(nwords_src, EMB_SIZE),
                               initializer=tf.initializers.random_uniform(0.25, 0.25))
embededSRC = tf.nn.embedding_lookup(embeddingSRC, encoder_src)
embededSRC = tf.expand_dims(embededSRC, 0)

embeddingTRG = tf.get_variable(name="embedding_trg", shape=(nwords_trg, EMB_SIZE),
                               initializer=tf.initializers.random_uniform(0.25, 0.25))
embededTRG = tf.nn.embedding_lookup(embeddingTRG, decoder_src)
embededTRG = tf.expand_dims(embededTRG, 0)

encoder = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)

_,  hidden = tf.nn.dynamic_rnn(cell=encoder, inputs=embededSRC, dtype=tf.float32)


decoder = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE)

output, _ = tf.nn.dynamic_rnn(cell=decoder, inputs=embededTRG, initial_state=hidden, dtype=tf.float32, scope="plain_decoder")
projection = tf.layers.dense(inputs=output, units=nwords_trg, use_bias=True)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reshape(projection, shape=(-1, nwords_trg)), labels=decoder_target)
optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())


def calc_loss(sent):
    src = sent[0]
    prev_target = sent[1][:-1]
    next_target = sent[1][1:]
    feed_encoder = {encoder_src: src,
                    decoder_src: prev_target,
                    decoder_target: next_target}
    my_loss, _ = session.run([loss, optimizer], feed_dict=feed_encoder)
    return np.sum(my_loss)


for ITER in range(10):
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_loss(sent)
        train_loss += my_loss
        train_words += len(sent)
        if (sent_id + 1) % 100 == 0:
            print("--finished %r sentences" % (sent_id + 1))
            break
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
        ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_loss(sent)
        dev_loss += my_loss
        dev_words += len(sent)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
        ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start))
