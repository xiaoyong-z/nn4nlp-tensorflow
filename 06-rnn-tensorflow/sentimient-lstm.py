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
    with open(filename, "r", encoding="utf-8") as f:
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
HID_SIZE = 80

word_tag = tf.placeholder(dtype=tf.int32, shape=[None])
tag_tensor = tf.placeholder(dtype=tf.int32, shape=[1, ntags])
embedding = tf.get_variable(name="embedding", shape=(nwords, EMB_SIZE), initializer=tf.initializers.random_uniform(0.25, 0.25))
embeded = tf.nn.embedding_lookup(embedding, word_tag)
embeded = tf.expand_dims(embeded, 0)


lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(HID_SIZE)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(HID_SIZE)

outputs ,_  = tf.nn.bidirectional_dynamic_rnn(cell_fw= lstm_fw_cell,
                                                cell_bw= lstm_bw_cell,
                                                inputs= embeded,
                                                dtype= tf.float32)
h = tf.concat(outputs, 2)[:,-1,:]
logits = tf.layers.dense(h, units=ntags, use_bias=True, kernel_initializer=tf.glorot_uniform_initializer())
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tag_tensor)
optimizer = tf.train.AdamOptimizer().minimize(loss)



session = tf.Session()
session.run(tf.global_variables_initializer())


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        tag_one_hot = np.zeros([1, ntags])
        tag_one_hot[0][tag] = 1
        feed_train = {word_tag: words,
                      tag_tensor: tag_one_hot}
        my_loss, _ = session.run([loss, optimizer], feed_dict=feed_train)
        train_loss += my_loss

    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (
        ITER, train_loss / len(train), time.time() - start))
    test_correct = 0.0
    for words, tag in dev:
        tag_one_hot = np.zeros([1, ntags])
        tag_one_hot[0][tag] = 1
        feed_dev = {word_tag: words,
                      tag_tensor: tag_one_hot}
        my_logits = session.run(logits, feed_dict=feed_dev)
        predict = np.argmax(my_logits)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))
