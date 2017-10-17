import os
import numpy as np
import tensorflow as tf
from glove import loadWordVectors
from imdb import loadReviewWordIds
from imdb import stringToWordIds

wordIds, wordVectors = loadWordVectors('data/glove.6B/glove.6B.50d')

# sentence = 'I like dogs'
# tokens = sentence.lower().split()
# vectorizedSentence = np.zeros(len(tokens), dtype='int32')
#
# for index, token in enumerate(tokens):
#     vectorizedSentence[index] = words.index(token)
#
# with tf.Session() as session:
#     print(tf.nn.embedding_lookup(wordVectors, vectorizedSentence).eval())

ids = loadReviewWordIds(wordIds)


