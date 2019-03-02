from collections import defaultdict
import math
import time
import random
import tensorflow as tf
import numpy as np

# The length of the n-gram
N = 2

# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already converted
#       into an easy-to-use format with "<unk>" symbols. If we were using other
#       data we would have to do pre-processing and consider how to choose
#       unknown words, etc.
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      yield [w2i[x] for x in line.strip().split(" ")]

# Read in the data
train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)

N = 2 # The length of the n-gram
EMB_SIZE = 128 # The size of the embedding
HID_SIZE = 128 # The size of the hidden layer

x = tf.placeholder(name="x", shape=[None, 2], dtype=tf.int32)
y_true = tf.placeholder(name="y_true", shape=[None, nwords], dtype=tf.int32)
keep_prob = tf.placeholder(name="dropout", dtype=tf.float32)

embedding = tf.get_variable(name="embedding", shape=[nwords, EMB_SIZE], initializer=tf.glorot_normal_initializer())
embeded = tf.nn.embedding_lookup(embedding, x)

embeded = tf.reshape(embeded, (-1, embeded.shape[1] * embeded.shape[2]))
dense1 = tf.layers.dense(embeded, name="dense1", units=HID_SIZE, activation=tf.tanh, use_bias=True, kernel_initializer=tf.glorot_normal_initializer())
dense1 = tf.nn.dropout(dense1, keep_prob=keep_prob)
dense2 = tf.layers.dense(name="dense2", inputs=dense1, units=nwords)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense2, labels=y_true)
optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for j in np.arange(10):
    random.shuffle(train)
    total_loss = 0
    train_words = 0
    for id, sentence in enumerate(train):
        total_history = []
        total_label = np.zeros([len(sentence) + 1, nwords])
        history = [S] * N
        sentence_loss = 0
        for i, char in enumerate(sentence + [S]):
            total_history.append(history)
            total_label[i][char] = 1
            history = history[1:] + [char]
        feed_train = {x: total_history,
                      y_true:total_label,
                      keep_prob:0.2}
        sentence_loss, _ = session.run(fetches=[loss, optimizer], feed_dict=feed_train)
        sentence_loss = np.sum(sentence_loss)
        total_loss += sentence_loss
        train_words += len(sentence)

        if (id + 1) % 5000 == 0:
            print("--finished %r sentences, %.4f" % (id + 1, (total_loss / train_words)))

    print("iter %r: train loss/word=%.4f, ppl=%.4f" % (
        i, total_loss / train_words, math.exp(total_loss / train_words)))
