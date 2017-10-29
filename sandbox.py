import os
from random import randint
import numpy as np
import tensorflow as tf
from glove import load_word_vectors
from imdb import load_dataset

word_ids, word_vectors = load_word_vectors('data/glove.6B/glove.6B.50d')

dataset = load_dataset(word_ids)

print(dataset[3]['review'])

with tf.Session() as session:
    print(tf.nn.embedding_lookup(word_vectors, dataset[3]['review']).eval())



