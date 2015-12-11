# __author__ = 'yanhe'


import numpy as np
import scipy.io as sp


def generalization_error(predictions, true_labels):
    # input: predictions = 1 by 2543 vector with predictions (0 = inactive; 1 = active)
    # output e = classification error
    return (np.count_nonzero(predictions == true_labels) + 0.0) / len(true_labels)


def test_error(predictions, true_labels):
    # input: predictions = 1 by 250 vector with predictions (0 = inactive; 1 = active)
    # output e = classification error
    true_labels = true_labels[0: 250]
    return (np.count_nonzero(predictions == true_labels) + 0.0) / len(true_labels)


# if __name__ == "__main__":
#     predictions = np.zeros(10)
#     generalization_error(predictions)
