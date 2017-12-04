import numpy as np
import tensorflow as tf
from glove import load_word_vectors
from imdb import clean_string, string_to_word_ids
from params import MAX_SEQUENCE_LENGTH, BATCH_SIZE, NUMBER_OF_CLASSES, \
    WORD_VECTOR_DIMENSION, NUMBER_OF_LSTM_UNITS, NUMBER_OF_LAYERS, NUMBER_OF_ITERATIONS

word_ids, word_vectors = load_word_vectors('data/glove.6B/glove.6B.50d')

input_data = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])

# data = tf.Variable(tf.zeros([BATCH_SIZE, MAX_SEQUENCE_LENGTH, WORD_VECTOR_DIMENSION]), dtype=tf.float32)
data = tf.nn.embedding_lookup(word_vectors, input_data)

def lstm_cell():
    lstmCell = tf.contrib.rnn.BasicLSTMCell(NUMBER_OF_LSTM_UNITS)
    return tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(NUMBER_OF_LAYERS)])

value, _ = tf.nn.dynamic_rnn(stacked_lstm, data, dtype=tf.float32)

# todo: make stacked lstms

weight = tf.Variable(tf.truncated_normal([NUMBER_OF_LSTM_UNITS, NUMBER_OF_CLASSES]), dtype=tf.float32)
bias = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]), dtype=tf.float32)

# todo: wtf is going on on these lines?
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)


prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)


sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))


input_text = clean_string("""
Best movie i've seen
""")
input_word_ids = string_to_word_ids(input_text, word_ids)

input_data_from_user = np.full([1, MAX_SEQUENCE_LENGTH], -1)
for i, word_id in enumerate(input_word_ids):
    if i >= MAX_SEQUENCE_LENGTH:
        break
    input_data_from_user[0][i] = input_word_ids[i]


for i in range(10):
    predictedSentiment = sess.run(prediction, {input_data: input_data_from_user})[0]
    if predictedSentiment[0] > predictedSentiment[1]:
        print("Positive Sentiment")
    else:
        print("Negative Sentiment")


# predictedSentiment[0] represents output score for positive sentiment
# predictedSentiment[1] represents output score for negative sentiment
