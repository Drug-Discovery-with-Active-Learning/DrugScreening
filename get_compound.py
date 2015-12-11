# __ author__ = 'yanhe'


import csv
import random
import numpy as np


def next_compound():
    features = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            features.append(cur)
    size = len(features)
    pick = random.sample(range(size), 1)[0]
    return features[pick]


# if __name__ == "__main__":
#     feature = next_compound()
#     print feature
