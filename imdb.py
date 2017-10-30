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


def reviews_to_dataset(word_ids, dirs):
    positive_reviews = [os.path.join(dirs['positive'], f) for f in os.listdir(dirs['positive'])]
    negative_reviews = [os.path.join(dirs['negative'], f) for f in os.listdir(dirs['negative'])]

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


def resize_data(data, max_sequence_length):
    for item in data:
        item['review'] = item['review'][:max_sequence_length]
        negatives = np.full(max_sequence_length - item['review'].shape[0], -1, dtype='int32')
        item['review'] = np.concatenate([item['review'], negatives])


def load_dataset(word_ids, force=False):
    train_path = 'data/aclImdb/train/train.npy'
    test_path = 'data/aclImdb/test/test.npy'

    if os.path.isfile(train_path) and os.path.isfile(test_path) and not force:
        train = np.load(train_path)
        test = np.load(test_path)
    else:
        train = reviews_to_dataset(word_ids, {'positive': 'data/aclImdb/train/pos',
                                              'negative': 'data/aclImdb/train/neg'})
        np.save(train_path, train)
        test = reviews_to_dataset(word_ids, {'positive': 'data/aclImdb/test/pos',
                                             'negative': 'data/aclImdb/test/neg'})
        np.save(test_path, test)

    return train, test
