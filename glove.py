import os
import numpy as np


"""Creates two binary files - word_ids.npy and word_vectors.npy - out of glove .txt file."""
def process_glove_txt(directory, file_name):
    word_ids = []
    word_vectors = []

    with open(os.path.join(directory, file_name), encoding='utf-8') as file_object:
        for line in file_object:
            tokens = line.split()
            word_ids.append(tokens[0])
            word_vectors.append(np.array(tokens[1:]).astype(np.float32))

    directory_to_save = os.path.join(directory, os.path.splitext(file_name)[0])
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save)

    np.save(os.path.join(directory_to_save, 'word_ids.npy'), word_ids)
    np.save(os.path.join(directory_to_save, 'word_vectors.npy'), word_vectors)


"""Returns a tuple of list of words and list of word vectors"""
def load_word_vectors(path):
    try:
        word_ids = np.load(os.path.join(path, 'word_ids.npy')).tolist()
        word_vectors = np.load(os.path.join(path, 'word_vectors.npy')) .astype(np.float32)
    except FileNotFoundError:
        print('.npy files were not found. Ensure you\'ve called process_glove_txt first.')
        raise

    return word_ids, word_vectors
