import os
from random import randint
import numpy as np
import tensorflow as tf
from glove import load_word_vectors
from imdb import load_dataset

word_ids, word_vectors = load_word_vectors('data/glove.6B/glove.6B.50d')

# sentence = 'I like dogs'
# tokens = sentence.lower().split()
# vectorizedSentence = np.zeros(len(tokens), dtype='int32')
#
# for index, token in enumerate(tokens):
#     vectorizedSentence[index] = words.index(token)
#
# with tf.Session() as session:
#     print(tf.nn.embedding_lookup(wordVectors, vectorizedSentence).eval())

dataset = load_dataset(word_ids)

print(dataset[24999])


