import os
import numpy as np


"""Creates two binary files - words.npy and wordVectors.npy - out of glove .txt file."""
def processGloveTxt(directory, fileName):
    words = []
    wordVectors = []

    with open(os.path.join(directory, fileName), encoding='utf-8') as fileObject:
        for line in fileObject:
            tokens = line.split()
            words.append(tokens[0])
            wordVectors.append(np.array(tokens[1:]).astype(np.float))

    directoryToSave = os.path.join(directory, os.path.splitext(fileName)[0])
    if not os.path.exists(directoryToSave):
        os.makedirs(directoryToSave)

    np.save(os.path.join(directoryToSave, 'wordIds.npy'), words)
    np.save(os.path.join(directoryToSave, 'wordVectors.npy'), wordVectors)


"""Returns a tuple of list of words and list of word vectors"""
def loadWordVectors(path):
    try:
        wordIds = np.load(os.path.join(path, 'wordIds.npy')).tolist()
        wordVectors = np.load(os.path.join(path, 'wordVectors.npy'))
    except FileNotFoundError:
        print('.npy files were not found. Ensure you\'ve called processGloveTxt first.')
        raise

    return wordIds, wordVectors
