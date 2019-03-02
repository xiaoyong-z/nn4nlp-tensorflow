from collections import defaultdict
import math
import time
import random
import tensorflow as tf
import numpy as np

N=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128 # The size of the embedding

embeddings_location = "embeddings.txt" #the file to write the word embeddings to
labels_location = "labels.txt" #the file to write the labels to

# We reuse the data reading from the language modeling class
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

with open(labels_location, 'w') as labels_file:
  for i in range(nwords):
    labels_file.write(i2w[i] + '\n')

x = tf.placeholder(dtype=tf.int32, shape=[None, 2*N])
y_true = tf.placeholder(dtype=tf.int32, shape=[None, nwords])

embedding = tf.get_variable(name="embedding", shape=[nwords, EMB_SIZE], initializer=tf.glorot_normal_initializer())
embeded = tf.nn.embedding_lookup(embedding, x)
embeded = tf.reduce_sum(embeded, axis=1)

output = tf.layers.dense(embeded, units=nwords, use_bias=False, kernel_initializer=tf.glorot_normal_initializer())

loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_true)

optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for i in range(100):
  random.shuffle(train)
  train_loss = 0
  train_words = 0
  for sent_id, sent in enumerate(train):
    train_words += len(train)
    if isinstance(sent, int):
        sent = [sent]
    train = [S]*N + sent + [S]*N
    input = []
    label = []
    for j in range(N, len(sent)+N):
      input.append(train[j-N:j] + train[j+1:j+N+1])
      y_one_hot = np.zeros([nwords])
      y_one_hot[train[j]] = 1
      label.append(y_one_hot)
    feed_train = {x:input, y_true:label}
    single_loss, _ = session.run([loss,optimizer], feed_dict=feed_train)
    single_loss = np.sum(single_loss)
    train_loss += single_loss
  print("iter %r: train loss/word=%.4f, ppl=%.4f" % (
  i, train_loss / train_words, math.exp(train_loss / train_words)))
session.close()