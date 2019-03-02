from collections import defaultdict
import time
import random
import tensorflow as tf
import numpy as np


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]


def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])


# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Define the model
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64
BATCH_SIZE = 12
PAD = 100

word_tag = tf.placeholder(dtype=tf.int32, shape=[None, PAD])
tag_tensor = tf.placeholder(dtype=tf.int32, shape=[None, ntags])
tag_true = tf.argmax(tag_tensor, axis=1)
embedding = tf.get_variable(name="embedding", shape=(nwords, EMB_SIZE))
embeded = tf.nn.embedding_lookup(embedding, word_tag) # BATCH_SIZE * PAD * TAGS
conv = tf.layers.conv1d(inputs=embeded, filters=FILTER_SIZE, kernel_size=WIN_SIZE, strides=1, padding="same", use_bias=True)
# conv : BATCH_SIZE * PAD * FILTER_SIZE
h = tf.reduce_max(conv, reduction_indices=[1])
relu = tf.nn.relu(h)
logits = tf.layers.dense(relu, units=ntags, use_bias=True,)

tag_predict = tf.argmax(logits, axis=1)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tag_tensor)
optimizer = tf.train.AdamOptimizer().minimize(loss)



session = tf.Session()
session.run(tf.global_variables_initializer())


def pad(words):
    pad_words = np.zeros([1, PAD])
    min_length = min(PAD, len(words))
    for i in range(min_length):
        pad_words[0][i] = words[i]
    return pad_words


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()

    words_batch = np.zeros([BATCH_SIZE, PAD])
    tag_one_hot_batch = np.zeros([BATCH_SIZE, ntags])
    count = 0
    for words, tag in train:
        if count == BATCH_SIZE:
            feed_train = {word_tag: words_batch,
                          tag_tensor: tag_one_hot_batch}
            predict, my_loss, _ = session.run([tag_predict, loss, optimizer], feed_dict=feed_train)
            for i in range(len(predict)):
                if predict[i] == (np.argmax(tag_one_hot_batch[i])):
                    train_correct += 1
            train_loss += np.sum(my_loss)
            tag_one_hot_batch = np.zeros([BATCH_SIZE, ntags])
            count = 0
        # Padding (can be done in the conv layer as well)
        if len(words) < WIN_SIZE:
            words += [0] * (WIN_SIZE - len(words))

        words_batch[count] = pad(words)
        tag_one_hot_batch[count][tag-1] = 1
        count = count + 1

#maybe the iteration above finish with count > 0, we still need to train with the half-batch
    feed_train = {word_tag: words_batch,
                  tag_tensor: tag_one_hot_batch}
    predict, my_loss, _ = session.run([tag_predict, loss, optimizer], feed_dict=feed_train)
    train_loss += np.sum(my_loss)
    for i in range(len(predict)):
        if predict[i] == (np.argmax(tag_one_hot_batch[i])):
            train_correct += 1


    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (
        ITER, train_loss / len(train), train_correct / len(train), time.time() - start))

