import os
from random import randint
import datetime
import numpy as np
import tensorflow as tf
from glove import load_word_vectors, process_glove_txt
from imdb import load_dataset
from imdb import resize_data
from params import MAX_SEQUENCE_LENGTH, BATCH_SIZE, NUMBER_OF_CLASSES, \
    WORD_VECTOR_DIMENSION, NUMBER_OF_LSTM_UNITS, NUMBER_OF_LAYERS, NUMBER_OF_ITERATIONS
from utils import getBatch


word_ids, word_vectors = load_word_vectors('data/glove.6B/glove.6B.50d')


train, test = load_dataset(word_ids)
resize_data(train, MAX_SEQUENCE_LENGTH)
resize_data(test, MAX_SEQUENCE_LENGTH)

labels = tf.placeholder(tf.int32, [None, NUMBER_OF_CLASSES])
input_data = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])

data = tf.nn.embedding_lookup(word_vectors, input_data)

def lstm_cell():
    lstmCell = tf.contrib.rnn.LSTMCell(NUMBER_OF_LSTM_UNITS)
    return tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(NUMBER_OF_LAYERS)])

value, _ = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([NUMBER_OF_LSTM_UNITS, NUMBER_OF_CLASSES]), dtype=tf.float32)
bias = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]), dtype=tf.float32)

value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


for i in range(NUMBER_OF_ITERATIONS):
    nextBatch, nextBatchLabels = getBatch(train, BATCH_SIZE, MAX_SEQUENCE_LENGTH)
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # Write summary to Tensorboard
    if i % 50 == 0:
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    # Save the network every 10,000 training iterations
    if i % 10000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)

writer.close()


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getBatch(test, BATCH_SIZE, MAX_SEQUENCE_LENGTH)
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
