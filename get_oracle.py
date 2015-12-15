# __author__ = 'yanhe'

import scipy.io as sio
import numpy as np
import csv


def read_mat():
    m = sio.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    return true_labels


# input: n is the compound number
# output: o the true label (0 = inactive; 1 = active)
def oracle1(true_labels, n):
    return true_labels[n]


# input: features is the feature vector returned by nextCompound
# output: label is the true label (0 = inactive; 1 = active)
def oracle2(point, data):
    size = len(data)
    m = sio.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    for n in range(0, size):
        if np.sum(data[n] == point) == len(point):
            label = true_labels[n]
            return label
