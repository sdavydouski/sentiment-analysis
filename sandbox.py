import os
from random import randint
import numpy as np
import tensorflow as tf
from glove import load_word_vectors
from imdb import load_dataset
from imdb import resize_data
from params import MAX_SEQUENCE_LENGTH, BATCH_SIZE
from utils import getBatch

word_ids, word_vectors = load_word_vectors('data/glove.6B/glove.6B.50d')

train, test = load_dataset(word_ids)
resize_data(train, MAX_SEQUENCE_LENGTH)
resize_data(test, MAX_SEQUENCE_LENGTH)

data, label = getBatch(train, BATCH_SIZE, MAX_SEQUENCE_LENGTH)

print(train[3])
print(test[5])

# with tf.Session() as session:
#     print(tf.nn.embedding_lookup(word_vectors, dataset[3]['review']).eval())



