# __author__ = 'Yan'


import csv
import numpy as np
import random
import numpy.ma as ma
import scipy.io as sio
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import DistanceMetric
from sklearn.svm import LinearSVC

import get_oracle as oracle
import get_error as err


from sklearn.metrics import jaccard_similarity_score as jaccard


def pool_reader():
    feature = []
    with open('resources/pool.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append(num)
    data = np.array(feature)
    return data


def rfc_learner():
    accuracy = []
    data = pool_reader()
    [row_size, col_size] = data.shape
    points = np.empty([0, col_size])
    labels = []
    used = set()
    cluster = k_means(data)
    cluster_zero = np.where(cluster == 0)[0]
    cluster_one = np.where(cluster == 1)[0]
    for i in xrange(0, 256):
        # if i == 0:
        #     pick = random.sample(range(row_size), 1)[0]
        #     used.add(pick)
        # else:
        #     # pick = get_next(data, points, used)
        #     pick = get_next_bool(data, points, used)
        #     used.add(pick)
        while True:
            if i % 50 == 0:
                pick = cluster_zero[random.sample(range(len(cluster_zero)), 1)[0]]
            else:
                pick = cluster_one[random.sample(range(len(cluster_one)), 1)[0]]
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
        accuracy.append(err.generalization_error(predictions))

    plt.plot(accuracy)
    plt.show()
    return accuracy


def k_means(data):
    clf = KMeans(n_clusters=2, copy_x=True)
    cluster = clf.fit_predict(data)
    return cluster


def get_next(data, active, used):
    if type(active) == int:
        score = []
        for x in xrange(data.shape[0]):
            score.append(jaccard(data[x], data[active]))
        rank = np.argsort(score)[::-1]
        for x in xrange(len(rank)):
            if rank[x] not in used:
                # print rank[x]
                # print jaccard(data[rank[x]], data[active])
                return rank[x]
    else: # type(active) == np.list
        score = []
        for x in xrange(data.shape[0]):
            cur_list = []
            for y in xrange(len(active)):
                cur_list.append(binary_vec_sim(data[x], data[y]))
            score.append(np.sum(cur_list))
        rank = np.argsort(score)
        for x in xrange(len(rank)):
            # if rank[x] not in used and score[rank[x]] != 0:
            if rank[x] not in used:
                # print rank[x]
                # print 'score:', score[rank[x]]
                return rank[x]


def svc_learner():
    accuracy = []
    data = pool_reader()
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
            accuracy.append(err.generalization_error(preds))
            if np.sum(labels) == 1 and len(labels) > 1:
                accuracy.pop()
                break

    X = np.array(selected)
    y = np.array(labels)

    clf = SVC(kernel = 'linear')
    # clf = LinearSVC()

    clf.fit(X, y)
    preds = clf.predict(data)
    accuracy.append(err.generalization_error(preds))

    for x in xrange(256-len(used)):
        # print x

        # random selection strategy
        # while 1:
        #     cur = random.randint(0, row-1)
        #     if cur not in used:
        #         break

        # farthest or say most different to previous 1 active selection strategy

        # active = np.where(y == 1)[0].tolist()

        # farthest to all used
        active = list(used)
        cur = get_next(data, active, used)
        print 'oracle', oracle.oracle1(cur)


        used.add(cur)
        X = np.vstack([X, data[cur]])
        y = np.hstack([y.tolist(),[oracle.oracle1(cur)]])
        clf.fit(X, y)
        preds = clf.predict(data)
        accuracy.append(err.generalization_error(preds))
        # print err.generalization_error(preds)

    plt.plot(accuracy)
    plt.show()
    return accuracy


def svc_margin_learner():
    accuracy = []
    data = pool_reader()
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
            accuracy.append(err.generalization_error(preds))
            if np.sum(labels) == 1 and len(labels) > 1:
                accuracy.pop()
                break

    X = np.array(selected)
    y = np.array(labels)

    clf = SVC(kernel = 'linear')
    # clf = LinearSVC()
    clf.fit(X, y)
    preds = clf.predict(data)
    accuracy.append(err.generalization_error(preds))

    for x in xrange(256-len(used)):
        # nearest to decision boundary
        distance = clf.decision_function(data)
        rank = np.argsort(np.abs(distance))
        for x in xrange(len(rank)):
            if rank[x] not in used:
                cur = rank[x]
                break

        print 'oracle', oracle.oracle1(cur)

        # closest to previous 1 active selection strategy
        # active = y.tolist().index(1)
        active = np.where(y == 1)[0].tolist()
        # print active
        cur = get_next(data, active, used)
        # print 'oracle',oracle.oracle1(cur)

        used.add(cur)
        X = np.vstack([X, data[cur]])
        y = np.hstack([y.tolist(),[oracle.oracle1(cur)]])
        clf.fit(X, y)
        preds = clf.predict(data)
        accuracy.append(err.generalization_error(preds))
        # print err.generalization_error(preds)

    plt.plot(accuracy)
    plt.show()
    return accuracy


def binary_vec_sim(a, b):
    return -np.count_nonzero(a-b)


if __name__ == "__main__":
    accuracy_vec = svc_margin_learner()
