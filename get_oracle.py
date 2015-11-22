# __author__ = 'Yan'

import scipy.io as sp
import numpy as np
import csv


# input: n is the compound number
# output: o the true label (0 = inactive; 1 = active)
def oracle1(n):
    m = sp.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    return true_labels[n]


# input: features is the feature vector returned by nextCompound
# output: label is the true label (0 = inactive; 1 = active)
def oracle2(features):
    data = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            data.append(cur)
    size = len(data)
    m = sp.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    for n in range(0, size):
        if np.sum(np.abs(np.subtract(data[n, :], features))) == 0:
            label = true_labels[n]
            return label
