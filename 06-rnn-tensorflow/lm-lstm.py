from __future__ import print_function
import time

start = time.time()

from collections import Counter, defaultdict
import random
import math
import sys
import argparse
import tensorflow as tf
import numpy as np

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

EMB_SIZE = 64
HID_SIZE = 128

word_tag = tf.placeholder(dtype=tf.int32, shape=[None])
tag_tensor =  tf.placeholder(dtype=tf.int32, shape=[None])
embedding = tf.get_variable(name="embedding", shape=(nwords, EMB_SIZE), initializer=tf.initializers.random_uniform(0.25, 0.25))
embeded = tf.nn.embedding_lookup(embedding, word_tag)
embeded = tf.expand_dims(embeded, 0)

lstm_cell = tf.nn.rnn_cell.LSTMCell(HID_SIZE)
initial_state = lstm_cell.zero_state(1, dtype=tf.float32)
outputs ,_  = tf.nn.dynamic_rnn(lstm_cell, embeded, initial_state=initial_state, dtype=tf.float32)
logits = tf.layers.dense(outputs, units=nwords, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer())
logits = tf.reduce_sum(logits, 0)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tag_tensor)
optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

def calc_lm_loss(sent, session, train):

    # get the wids and masks for each step
    totWords = len(sent)
    sent = [S] + sent
    feed = {word_tag: sent,
            tag_tensor: sent}
    if train:
        my_loss, _ = session.run([loss, optimizer], feed_dict=feed)
    else:
        my_loss = session.run([loss], feed_dict=feed)
    return my_loss, totWords


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
                loss_exp, mb_words = calc_lm_loss(sent, session, False)
                dev_loss += np.sum(loss_exp)
                dev_words += mb_words
            dev_time += time.time() - dev_start
            train_time = time.time() - start - dev_time
            print("nll=%.4f, ppl=%.4f, words=%r, time=%.4f, word_per_sec=%.4f" % (
            dev_loss / dev_words, math.exp(dev_loss / dev_words), dev_words, train_time, all_tagged / train_time))
        # train on the minibatch
        loss_exp, mb_words = calc_lm_loss(train[sid], session, True)
        this_loss += np.sum(loss_exp)
        this_words += mb_words
    print("epoch %r finished" % ITER)
session.close()