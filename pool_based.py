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


def norm_learner():
    accuracy = []
    data = pool_reader()
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
        accuracy.append(err.generalization_error(predictions))

    plt.plot(accuracy)
    plt.show()
    return accuracy


def lrc_learner():
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
        while True:
            pick = cluster_one[random.sample(range(len(cluster_one)), 1)[0]]
            if pick not in used:
                break
        used.add(pick)
        points = np.vstack([points, data[pick]])
        labels.append(oracle.oracle1(pick))
        if i >= 20:
            clf = LogisticRegression(penalty='l2')
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


# def get_next(data, points, used):
#     dist = distance.cdist(data, points, 'euclidean')
#     sum_dist = np.sum(dist, axis=1)
#     rank = np.argsort(sum_dist)[::-1][:len(sum_dist)]
#     for i in xrange(0, len(rank)):
#         if rank[i] not in used:
#             return i


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
            print sum_dist[i]
            return i


def getNext(data, active, used):
    if type(active) == int:
        score = []
        for x in xrange(data.shape[0]):
            score.append(jaccard(data[x], data[active]))
        rank = np.argsort(score)[::-1]
        for x in xrange(len(rank)):
            if rank[x] not in used:
                #print rank[x]
                #print jaccard(data[rank[x]], data[active])
                return rank[x]
    else: # type(active) == np.list
        score = []
        for x in xrange(data.shape[0]):
            cur_list = []
            for y in xrange(len(active)):
                cur_list.append(binaryVecSim(data[x], data[y]))
            score.append(np.sum(cur_list))
        rank = np.argsort(score)
        for x in xrange(len(rank)):
            #if rank[x] not in used and score[rank[x]] != 0:
            if rank[x] not in used:
                #print rank[x]
                #print 'score:', score[rank[x]]
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
    #clf = LinearSVC()
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
        cur = getNext(data, active, used)
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

def svc_learner_new():
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
    #clf = LinearSVC()
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



def binaryVecSim(a, b):
    return -np.count_nonzero(a-b)

if __name__ == "__main__":
    accuracy_vec = svc_learner_new()
