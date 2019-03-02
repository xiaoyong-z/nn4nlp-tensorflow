from collections import defaultdict
import time
import random
import numpy as np
from datetime import timedelta
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
tf.set_random_seed(100)

# Functions to read in the corpus
def read_x(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield words

def read_y(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield int(tag)



train_y = list(read_y("../data/classes/train.txt"))
train_x = list(read_x("../data/classes/train.txt"))
dev_y = list(read_y("../data/classes/test.txt"))
dev_x = list(read_x("../data/classes/test.txt"))

all_reviews = train_x + dev_x
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_reviews)
print("Fitting is complete.")

train_x = tokenizer.texts_to_sequences(train_x)
print("train_seq is complete.")

dev_x = tokenizer.texts_to_sequences(dev_x)
print("test_seq is complete")




ntags = 5
word_index = tokenizer.word_index
nwords = len(word_index) + 1
x_s = tf.placeholder(dtype=tf.int32, shape=[None])
y_true = tf.placeholder(dtype=tf.int32, shape=[1, ntags])
#keep_prob = tf.placeholder(dtype=tf.float32)

embedding = tf.Variable(tf.glorot_normal_initializer(dtype=tf.float32)((nwords, ntags)))
embed = tf.nn.embedding_lookup(embedding, x_s)
b = tf.Variable(tf.constant(0, shape=[ntags], dtype=tf.float32))


logits = tf.reduce_sum(embed, axis=0) + b
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)
optimizer = tf.train.AdamOptimizer().minimize(loss)

session = tf.Session()
session.run(tf.initialize_all_variables())



def optimize(times):
    for i in range(times):
        train_loss = 0
        for j in np.arange(len(train_x)):
            x_train = train_x[j]
            y_train = train_y[j]
            y_one_hot = np.zeros([1, 5])

            y_one_hot[0][y_train] = 1
            feed_train = {x_s: x_train, y_true: y_one_hot}
            batch_loss, _ = session.run([loss, optimizer], feed_dict=feed_train)
            train_loss += batch_loss


        msg = "Iteration: {0:>6}, Train loss: {1:>6.5}"
        print(msg.format(i, train_loss/len(train_x)))

        acc = 0
        for j in np.arange(len(dev_x)):
            x_dev = dev_x[j]
            y_dev = dev_y[j]
            y_one_hot = np.zeros([1, 5])

            y_one_hot[0][y_dev] = 1
            feed_train = {x_s: x_dev, y_true: y_one_hot}
            dev_logits = session.run(logits, feed_dict=feed_train)
            result = np.argmax(dev_logits)
            if result == y_dev:
                acc += 1

        msg = "Iteration: {0:>6}, Test acc: {1:>6.5}"
        print(msg.format(i, acc/len(dev_x)))


optimize(1000)