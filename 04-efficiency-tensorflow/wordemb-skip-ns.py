from collections import defaultdict
import math
import numpy as np
import time
import random
import tensorflow as tf


K=3 #number of negative samples
N=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128 # The size of the embedding

embeddings_location = "embeddings.txt" #the file to write the word embeddings to
labels_location = "labels.txt" #the file to write the labels to

# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))

#word counts for negative sampling
word_counts = defaultdict(int)

S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      line = line.strip().split(" ")
      for word in line:
        word_counts[w2i[word]] += 1
      yield [w2i[x] for x in line]


# Read in the data
train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)


# take the word counts to the 3/4, normalize
counts =  np.array([list(x) for x in word_counts.items()])[:,1]**.75
normalizing_constant = sum(counts)
word_probabilities = np.zeros(nwords)
for word_id in word_counts:
  word_probabilities[word_id] = word_counts[word_id]**.75/normalizing_constant

with open(labels_location, 'w') as labels_file:
  for i in range(nwords):
    labels_file.write(i2w[i] + '\n')






word_position = tf.placeholder(dtype=tf.int32, shape=[None])
positive_context_position = tf.placeholder(dtype=tf.int32, shape=[None, None])
negative_context_position = tf.placeholder(dtype=tf.int32, shape=[None, None])

word_embedding = tf.get_variable(name="word_embedding", shape=(nwords, EMB_SIZE), initializer=tf.glorot_normal_initializer())
context_embedding = tf.get_variable(name="context_embedding", shape=(nwords, EMB_SIZE), initializer=tf.glorot_normal_initializer())

embed_word = tf.nn.embedding_lookup(word_embedding, word_position)
positive_embed_context = tf.nn.embedding_lookup(context_embedding, positive_context_position)
negative_embed_context = tf.nn.embedding_lookup(context_embedding, negative_context_position)
embed_word = tf.expand_dims(embed_word, 2)
positive_score = tf.matmul(positive_embed_context, embed_word)
negative_score = -tf.matmul(negative_embed_context, embed_word)
positive_loss = tf.reduce_sum(-tf.reduce_sum(tf.log_sigmoid(positive_score), axis=1))
negative_loss = tf.reduce_sum(-tf.reduce_sum(tf.log_sigmoid(negative_score), axis=1))
loss = positive_loss + negative_loss

optimizer = tf.train.AdamOptimizer().minimize(loss)


session = tf.Session()
session.run(tf.global_variables_initializer())
def calc_sent_loss(sent, session):
    # add padding to the sentence equal to the size of the window
    # as we need to predict the eos as well, the future window at that point is N past it
    all_neg_words = np.random.choice(nwords, size=2*N*K*len(sent), replace=True, p=word_probabilities)

    # Step through the sentence
    losses = []
    word_batch = []
    pos_words_batch = []
    neg_words_batch = []
    for i, word in enumerate(sent):
        pos_words = [sent[x] if x >= 0 else S for x in range(i-N,i)] + \
                     [sent[x] if x < len(sent) else S for x in range(i+1,i+N+1)]
        neg_words = all_neg_words[i * K * 2 * N:(i + 1) * K * 2 * N]
        word_batch.append(word)
        pos_words_batch.append(pos_words)
        neg_words_batch.append(neg_words)
    feed_train = {word_position: word_batch,
                positive_context_position: pos_words_batch,
                negative_context_position: neg_words_batch}
    simple_loss, _ = session.run([loss, optimizer], feed_dict= feed_train)
    losses.append(simple_loss)
    return np.sum(losses)


MAX_LEN = 100

for ITER in range(100):
    print("started iter %r" % ITER)
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent, session)
        train_loss += my_loss.item()
        train_words += len(sent)
        # Back prop while training

        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
            train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
            print("after sentences %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
			sent_id + 1, train_loss / train_words, train_ppl, time.time() - start))
    train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, train_ppl, time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_sent_loss(sent, session)
        dev_loss += my_loss.item()
        dev_words += len(sent)
    dev_ppl = float('inf') if dev_loss / dev_words > 709 else math.exp(dev_loss / dev_words)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, dev_ppl, time.time() - start))

    print("saving embedding files")
    with open(embeddings_location, 'w') as embeddings_file:
        word_embedding = session.run(word_embedding)
        W_w_np = np.array(word_embedding.data)
        for i in range(nwords):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')

session.close()