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
        # if i == 0:
        #     pick = random.sample(range(row_size), 1)[0]
        #     used.add(pick)
        # else:
        #     # pick = get_next(data, points, used)
        #     pick = get_next_bool(data, points, used)
        #     used.add(pick)
        while True:
            pick = random.sample(range(row_size), 1)[0]
            if pick not in used:
                break
        used.add(pick)
        points = np.vstack([points, data[pick]])
        if oracle.oracle1(pick) == 1:
            print i, 'th iteration cur label ', 1, '\n'
        labels.append(oracle.oracle1(pick))
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(points, np.array(labels))
        predictions = clf.predict(data)
        loss.append(err.generalization_error(predictions))

    plt.plot(loss_vec)
    plt.show()
    return loss


def pool_leaner_norm(loss):
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            # TODO: changed num to norm
            feature.append([np.linalg.norm(num)])
    data = np.array(feature)
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
    rank = np.argsort(sum_dist)
    for i in xrange(0, len(rank)):
        if rank[i] not in used:
            return i





def svc_learner():
    loss = []
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append(num)
    data = np.array(feature)
    [row, col] = data.shape

    # do nothing about model until reasonable training subset achieved
    active_count = 0
    preds = np.zeros(row)
    used = set()
    selected = []
    labels = []
    while 1:
        r = random.randint(0, row-1)
        if r not in used:
            used.add(r)
            selected.append(data[r].tolist())
            labels.append(oracle.oracle1(r))
            used.add(r)
            loss.append(err.generalization_error(preds))
            if np.sum(labels) == 1:
                loss.pop()
                break


    X = np.array(selected)
    y = np.array(labels)
    #clf = SVC(kernel = 'linear', class_weight = {0:0.1, 1:0.9}, C = 0.1)
    clf = SVC(kernel = 'linear', class_weight = 'balanced', C = 0.1)
    clf.fit(X, y)
    preds = clf.predict(data)
    loss.append(err.generalization_error(preds))

    for x in xrange(256-len(used)):
        #cur = get_next_bool(data, X, used)
        cur = random.randint(0, row-1)
        if cur not in used:
            used.add(cur)
            X = np.vstack([X, data[cur]])
            y = np.hstack([y.tolist(),[oracle.oracle1(cur)]])
            clf.fit(X, y)
            preds = clf.predict(data)
            loss.append(err.generalization_error(preds))
            #print err.generalization_error(preds)
    plt.plot(loss)
    plt.show()
    return loss




if __name__ == "__main__":

    loss_vec = svc_learner()
