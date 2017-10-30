from random import randint
import numpy as np


def getBatch(dataset, batch_size, max_sequence_length):
    labels = []
    data = np.zeros([batch_size, max_sequence_length])
    for i in range(batch_size):
        if i % 2 == 0:
            index = randint(0, 12499)
        else:
            index = randint(12500, 24999)
        data[i] = dataset[index]['review']
        labels.append(dataset[index]['label'])

    return data, labels

