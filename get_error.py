# __author__ = 'Yan'


import numpy as np
import scipy.io as sp


def generalization_error(predictions):
    # input: predictions = 1 by 2543 vector with predictions (0 = inactive; 1 = active)
    # output e = classification error
    m = sp.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    return (np.count_nonzero(predictions == true_labels) + 0.0) / len(true_labels)


def test_error(predictions):
    # input: predictions = 1 by 250 vector with predictions (0 = inactive; 1 = active)
    # output e = classification error
    m = sp.loadmat('resources/trueLabels.mat')
    true_labels = np.array(m['trueLabels'][0])
    true_labels = true_labels[0: 250]
    return (np.count_nonzero(predictions == true_labels) + 0.0) / len(true_labels)


if __name__ == "__main__":
    predictions = np.zeros(10)
    generalization_error(predictions)
