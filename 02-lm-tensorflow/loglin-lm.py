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

x1 = tf.placeholder(shape=(1,), dtype=tf.int32)
x2 = tf.placeholder(shape=(1,), dtype=tf.int32)
y = tf.placeholder(shape=(1,None), dtype=tf.int32)
embedding1 = tf.get_variable(name="embedding1", shape=(nwords, nwords), initializer=tf.glorot_normal_initializer())
embedding2 = tf.get_variable(name="embedding2",shape=(nwords, nwords), initializer=tf.glorot_normal_initializer())
bias = tf.get_variable(name="bias", shape=(nwords), initializer=tf.glorot_normal_initializer())

embed1 = tf.nn.embedding_lookup(embedding1, x1)
embed2 = tf.nn.embedding_lookup(embedding2, x2)
score = embed1 + embed2 + bias

loss = tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y)

optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(10):
    random.shuffle(train)
    total_loss = 0
    train_words = 0
    for id, sentence in enumerate(train):
        history = [S] * N
        sentence_loss = 0
        for i in sentence + [S]:
            y_one_hot = np.zeros(shape=(1, nwords))
            y_one_hot[0][i] = 1
            input1, input2 = history
            history = history[1:] + [nwords]
            feed_train = {x1: [input1],
                          x2: [input2],
                          y: y_one_hot}
            char_loss, _  = session.run(fetches=[loss, optimizer], feed_dict=feed_train)
            sentence_loss += char_loss
        total_loss += sentence_loss
        train_words += len(sentence)

        if (id + 1) % 5000 == 0:
            print("--finished %r sentences, %.4f" % (id + 1, (total_loss / train_words)))

    print("iter %r: train loss/word=%.4f, ppl=%.4f" % (
    i, total_loss / train_words, math.exp(total_loss / train_words)))
