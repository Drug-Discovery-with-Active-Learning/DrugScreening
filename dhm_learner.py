# __author__ = 'Yan'


import numpy as np
import csv


# TODO: current using Matlab code
def dhm():
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append([np.linalg.norm(num)])
    # TODO: need to normalize
    data = np.array(feature)
    size = len(data)

    # DHM algorithm part initialization
    s_point = np.zeros(1, size)
    t_point = np.zeros(1, size)
    s_label = np.zeros(1, size)
    t_label = np.zeros(1, size)
    # r vector indicate which point has been queried
    r_vec = np.zeros(1, size)

    # DHM loop part
    for i in xrange(0, size):
        print i, '\n'
        t_cur = data[t_point == 1]
        t_label_cur = t_label[t_point == 1]

        # hypothesis label 1
        s_one = s_point
        s_one[i] = 1
        s_one_label = s_label
        s_one_label[i] = 1

        s_one_cur = data[s_one == 1]
        # t_cur = data[t_point == 1]
        s_one_label_cur = s_one_label[s_one == 1]
        # t_label_cur = t_label[t_point == 1]
        # TODO
        [h_one, flag_one] = dhm_learner(s_one_cur, t_cur, s_one_label_cur, t_label_cur)
        err_one = dhm_error(h_one, data[s_one + t_point == 1], s_one_label + t_label)

        # hypothesis label 0
        s_zero = s_point
        s_zero[i] = 1
        s_zero_label = s_label
        s_zero_label[i] = 0

        s_zero_cur = data[s_zero == 1]
        s_zero_label_cur = s_zero_label[s_zero == 1]
        [h_zero, flag_zero] = dhm_learner(s_zero_cur, t_cur, s_zero_label_cur, t_label_cur)
        err_zero = dhm_error(h_zero, data[s_zero + t_point == 1], s_zero_label + t_label)

        # calculate delta
        delta = 0.01
        shatter_coeff = 2 * (i + 2)


def dhm_learner(s_one_cur, t_cur, s_one_label_cur, t_label_cur):

    return 0, 0


def dhm_error(h, x, y):

    return 0
