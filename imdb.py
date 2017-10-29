import os
import re
import numpy as np


"""Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters"""
def clean_string(string):
    return re.sub('[^A-Za-z0-9 ]+', '', string.lower())


"""Transforms string into a list of word ids

Example:
'i like dogs' -> [41, 117, 3876]"""
def string_to_word_ids(string, word_ids):
    string_word_ids = []

    words = string.split()
    for word in words:
        try:
            string_word_ids.append(word_ids.index(word))
        except ValueError:
            print('Word not found', word)

    return string_word_ids


def reviews_to_dataset(word_ids):
    positive_directory = 'data/aclImdb/train/pos'
    negative_directory = 'data/aclImdb/train/neg'

    positive_reviews = [os.path.join(positive_directory, f) for f in os.listdir(positive_directory)]
    negative_reviews = [os.path.join(negative_directory, f) for f in os.listdir(negative_directory)]

    dataset = []

    for review in positive_reviews + negative_reviews:
        with open(review, 'r', encoding='utf-8') as file_object:
            review_string = ''
            for line in file_object:
                review_string += line

            review_string = clean_string(review_string)
            review_word_ids = string_to_word_ids(review_string, word_ids)
            dataset.append({
                'review': np.array(review_word_ids, dtype='int32'),
                'label': np.array([1, 0] if review in positive_reviews else [0, 1], dtype='int32')
            })

    return dataset


def resize_dataset(dataset, max_words):
    for item in dataset:
        item['review'] = item['review'][:max_words]
        negatives = np.full(max_words - item['review'].shape[0], -1, dtype='int32')
        item['review'] = np.concatenate([item['review'], negatives])


def load_dataset(word_ids, max_words=300, force=False):
    path = 'data/aclImdb/train/dataset.npy'

    if os.path.isfile(path) and not force:
        dataset = np.load(path)
    else:
        dataset = reviews_to_dataset(word_ids)
        np.save(path, dataset)

    resize_dataset(dataset, max_words)
    
    return dataset
