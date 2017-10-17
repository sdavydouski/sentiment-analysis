import os
import re
import numpy as np


"""Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters"""
def cleanLine(line):
    return re.sub('[^A-Za-z0-9 ]+', '', line.lower())


"""Transforms string into a list of word ids

Example:
'i like dogs' -> [41, 117, 3876]"""
def stringToWordIds(string, wordIds):
    stringWordIds = []

    words = string.split()
    for word in words:
        try:
            stringWordIds.append(wordIds.index(word))
        except ValueError:
            print('Word not found', word)

    return stringWordIds

"""Iterates through all positive and negative movie reviews and 
transforms them into a list of word ids"""
def reviewsToWordIds(wordIds):
    positiveDirectory = 'data/aclImdb/train/pos'
    negativeDirectory = 'data/aclImdb/train/neg'

    positiveReviews = [os.path.join(positiveDirectory, f) for f in os.listdir(positiveDirectory)]
    negativeReviews = [os.path.join(negativeDirectory, f) for f in os.listdir(negativeDirectory)]

    allReviews = positiveReviews + negativeReviews

    reviewsWordIds = []

    for review in allReviews:
        with open(review, 'r', encoding='utf-8') as fileObject:
            for line in fileObject:
                reviewsWordIds.append(np.array(stringToWordIds(cleanLine(line), wordIds), dtype='int32'))

    return reviewsWordIds

"""Returns a list of all movie review word ids either from pre-saved .npy file or 
by creating them from scratch"""
def loadReviewWordIds(wordIds, force=False):
    path = 'data/aclImdb/train/ids.npy'

    if os.path.isfile(path) and not force:
        ids = np.load(path)
    else:
        ids = reviewsToWordIds(wordIds)
        np.save(path, ids)

    return ids
