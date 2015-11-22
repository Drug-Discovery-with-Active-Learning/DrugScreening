# __author__ = 'Yan'


import csv
import numpy as np
import random
import scipy.spatial.distance as distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import get_oracle as oracle
import get_error as err
from sklearn.neighbors import DistanceMetric

from sklearn.metrics import jaccard_similarity_score as jaccard

def pool_leaner(loss):
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append(num)
    data = np.array(feature)
    [row_size, col_size] = data.shape
    points = np.empty([0, col_size])
    labels = []
    used = set()
    for i in xrange(0, 256):
        print i, '\n'
        if i == 0:
            pick = random.sample(range(row_size), 1)[0]
            used.add(pick)
        else:
            #pick = get_next(data, points, used)
            pick = get_next_bool(data, points, used)
            used.add(pick)
        points = np.vstack([points, data[pick]])
        labels.append(oracle.oracle1(pick))
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(points, np.array(labels))
        predictions = clf.predict(data)
        loss.append(err.generalization_error(predictions))

    plt.plot(loss_vec)
    plt.show()
    return loss


def pool_leaner2(loss):
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            # TODO: changed num
            feature.append([np.linalg.norm(num)])
    data = np.array(feature)
    # [row_size, col_size] = data.shape
    row_size = len(data)
    points = []
    labels = []
    used = set()
    for i in xrange(0, 256):
        print i, '\n'
        if i == 0:
            pick = random.sample(range(row_size), 1)[0]
            used.add(pick)
        else:
            pick = get_next(data, points, used)
            used.add(pick)
        points.append(data[pick])
        labels.append(oracle.oracle1(pick))
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(np.array(points), np.array(labels))
        predictions = clf.predict(data)
        loss.append(err.generalization_error(predictions))

    plt.plot(loss_vec)
    plt.show()
    return loss


def get_next(data, points, used):
    dist = distance.cdist(data, points, 'euclidean')
    sum_dist = np.sum(dist, axis=1)
    rank = np.argsort(sum_dist)[::-1][:len(sum_dist)]
    for i in xrange(0, len(rank)):
        if rank[i] not in used:
            return i

def get_next_bool(data, points, used):
    sum_dist = np.zeros(data.shape[0])
    for i in xrange(data.shape[0]):
        cur_list = []
        for j in xrange(points.shape[0]):
            cur_list.append(jaccard(data[i], points[j]))
        sum_dist[i] = np.sum(cur_list)
    rank = np.argsort(sum_dist)[::-1][:len(sum_dist)]
    for i in xrange(0, len(rank)):
        if rank[i] not in used:
            return i


if __name__ == "__main__":
    loss_vec = []
    loss_vec = pool_leaner(loss_vec)
