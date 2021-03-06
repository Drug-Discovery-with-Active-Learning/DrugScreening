# __author__ = 'yanhe' and 'xiaoxul'

import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score

import get_oracle as oracle
import get_error as err


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


def rfc_learner(option):
    accuracy = []
    data = pool_reader()
    true_labels = oracle.read_mat()
    [row_size, col_size] = data.shape
    points = np.empty([0, col_size])
    labels = []
    used = set()
    flag = True
    predictions = np.zeros(row_size)
    for i in xrange(0, 256):
        if option == 'select':
            if flag:
                pick = random.sample(range(row_size), 1)[0]
            else:
                # pick = get_next(data, points, used)
                clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
                clf.fit(points, np.array(labels))
                prob = clf.predict_proba(data)
                weight = np.abs(prob[:, 0] - 0.5)
                rank = np.argsort(weight)
                for x in xrange(len(rank)):
                    if rank[x] not in used:
                        pick = rank[x]
                        break
        else:
            while 1:
                pick = random.sample(range(row_size), 1)[0]
                if pick not in used:
                    break

        used.add(pick)
        points = np.vstack([points, data[pick]])
        if oracle.oracle1(true_labels, pick) == 1:
            flag = False
        labels.append(oracle.oracle1(true_labels, pick))
        clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
        clf.fit(points, np.array(labels))
        predictions = clf.predict(data)
        cur_acc = err.generalization_error(predictions, true_labels)
        accuracy.append(cur_acc)
    plt.plot(accuracy)
    plt.show()
    print "f1 ", f1_score(predictions, true_labels)
    return accuracy


def lrc_learner(option):
    accuracy = []
    data = pool_reader()
    true_labels = oracle.read_mat()
    [row_size, col_size] = data.shape
    predictions = np.zeros(row_size)
    points = np.empty([0, col_size])
    labels = []
    used = set()
    flag = True
    for i in xrange(0, 256):
        if option == "select":
            pick = -1
            if flag:
                while 1:
                    pick = random.sample(range(row_size), 1)[0]
                    if pick not in used:
                        used.add(pick)
                        points = np.vstack([points, data[pick]])
                        label = oracle.oracle1(true_labels, pick)
                        labels.append(label)
                        if label == 1:
                            flag = False
                        break
            else:
                clf = LogisticRegression()
                clf.fit(points, np.array(labels))
                prob = clf.predict_proba(data)
                weight = np.abs(prob[:, 0] - 0.5)
                rank = np.argsort(weight)
                for x in xrange(len(rank)):
                    if rank[x] not in used:
                        pick = rank[x]
                        break
                used.add(pick)
                points = np.vstack([points, data[pick]])
                label = oracle.oracle1(true_labels, pick)
                labels.append(label)
                clf.fit(points, np.array(labels))
                predictions = clf.predict(data)
                cur_acc = err.generalization_error(predictions, true_labels)
                accuracy.append(cur_acc)
        else:
            while 1:
                pick = random.sample(range(row_size), 1)[0]
                if pick not in used:
                    break
            used.add(pick)
            points = np.vstack([points, data[pick]])
            label = oracle.oracle1(true_labels, pick)
            labels.append(label)
            if label == 1:
                flag = False
            if not flag:
                clf = LogisticRegression()
                clf.fit(points, np.array(labels))
                predictions = clf.predict(data)
                cur_acc = err.generalization_error(predictions, true_labels)
                accuracy.append(cur_acc)
    plt.plot(accuracy)
    plt.show()
    print "f1 ", f1_score(predictions, true_labels)
    return accuracy


def get_next(data, active, used):
    score = []
    for x in xrange(data.shape[0]):
        cur_list = []
        for y in xrange(len(active)):
            cur_list.append(binary_vec_sim(data[x], data[y]))
        score.append(np.sum(cur_list))
    rank = np.argsort(score)
    for x in xrange(len(rank)):
        if rank[x] not in used:
            return rank[x]


def svm_learner(option):
    accuracy = []
    data = pool_reader()
    [row, col] = data.shape
    true_labels = oracle.read_mat()

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
            labels.append(true_labels[r])
            used.add(r)
            accuracy.append(err.generalization_error(preds, true_labels))
            if np.sum(labels) == 1 and len(labels) > 1:
                accuracy.pop()
                break

    X = np.array(selected)
    y = np.array(labels)
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    preds = clf.predict(data)
    accuracy.append(err.generalization_error(preds, true_labels))
    for x in xrange(256-len(used)):
        if option == 'rand':
            # random selection strategy
            while 1:
                cur = random.randint(0, row-1)
                if cur not in used:
                    break
        else:
            # farthest or say most different to previous 1 active selection strategy
            active = np.where(y == 0)[0].tolist()
            # farthest to all used
            cur = get_next(data, active, used)
            print 'oracle', true_labels[cur]
        used.add(cur)
        X = np.vstack([X, data[cur]])
        y = np.hstack([y.tolist(),[true_labels[cur]]])
        clf.fit(X, y)
        preds = clf.predict(data)
        accuracy.append(err.generalization_error(preds, true_labels))

    print f1_score(preds, true_labels)
    return accuracy


def svm_margin_learner():
    accuracy = []
    data = pool_reader()
    [row, col] = data.shape
    true_labels = oracle.read_mat()

    # do nothing about model until reasonable training subset achieved
    preds = np.zeros(row)
    used = set()
    selected = []
    labels = []
    while 1:
        r = random.randint(0, row-1)
        if r not in used:
            used.add(r)
            selected.append(data[r].tolist())
            labels.append(true_labels[r])
            accuracy.append(err.generalization_error(preds, true_labels))
            if np.sum(labels) == 1 and len(labels) > 1:
                accuracy.pop()
                break

    X = np.array(selected)
    y = np.array(labels)

    clf = SVC(kernel='linear')
    clf.fit(X, y)
    preds = clf.predict(data)
    accuracy.append(err.generalization_error(preds, true_labels))
    for x in xrange(256-len(used)):
        # nearest to decision boundary
        distance = clf.decision_function(data)
        rank = np.argsort(np.abs(distance))
        for i in xrange(len(rank)):
            if rank[i] not in used:
                cur = rank[i]
                break
        print 'oracle', true_labels[cur]
        used.add(cur)
        X = np.vstack([X, data[cur]])
        y = np.hstack([y.tolist(),[true_labels[cur]]])
        clf.fit(X, y)
        preds = clf.predict(data)
        accuracy.append(err.generalization_error(preds, true_labels))
    print f1_score(preds, true_labels)
    return accuracy


def binary_vec_sim(a, b):
    return -np.count_nonzero(a-b)


def svm_learner_all():
    data = pool_reader()
    true_labels = oracle.read_mat()
    clf = SVC(kernel='linear')
    X = np.array(data)
    y = np.array(true_labels)
    clf.fit(X, y)
    preds = clf.predict(data)
    accuracy = (err.generalization_error(preds, true_labels))
    print accuracy
    print f1_score(preds, true_labels)
    return accuracy


def precision(preds, true_labels):
    true_positive = len(np.where(preds+true_labels == 2)[0])
    selected = len(np.where(preds == 1)[0])
    return (true_positive+0.0)/selected


def recall(preds, true_labels):
    true_positive = len(np.where(preds+true_labels == 2)[0])
    relevant = len(np.where(true_labels == 1)[0])
    return (true_positive+0.0)/relevant


def f1_score(preds, true_labels):
    mcc = matthews_corrcoef(preds, true_labels)
    print "mcc ", mcc
    kappa = cohen_kappa_score(preds, true_labels)
    print "kappa ", kappa
    p = precision(preds, true_labels)
    print "precision ", p
    r = recall(preds, true_labels)
    print "recall", r
    return 2*p*r/(p+r)


if __name__ == "__main__":
    acc = rfc_learner('rand')
    # accuracy_vec = svm_margin_learner()
    # accuracy_vec = svm_learner('select')
