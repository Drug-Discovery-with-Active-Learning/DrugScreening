# author = 'yanhe' and 'xiaoxul'


import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score

import get_compound as compound
import get_oracle as oracle
import get_error as error


def csv_reader(path):
    feature = []
    with open(path, 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append(num)
    data = np.array(feature)
    return data


def stream_learner(method, option, budget):
    features = csv_reader('resources/pool.csv')
    [row, col] = features.shape
    testset = csv_reader('resources/testSet.csv')
    true_labels = oracle.read_mat()
    if method == "rf":
        clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
    if method == "lr":
        clf = LogisticRegression(penalty='l2')
    accuracy = []
    points = []
    labels = []
    used = {}
    flag = True
    query_count = 0
    i = 0
    pred = np.zeros(250)
    if option == "select":
        # active learner
        while i < 2543 and query_count < budget:
            if flag:
                # call next compound until get one point with label 1
                cur_point = compound.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str[1: (len(cur_str) - 1)] not in used:
                    i += 1
                    points.append(cur_point)
                    cur_label = oracle.oracle2(cur_point, features)
                    labels.append(cur_label)
                    used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                    query_count += 1
                    if cur_label == 1:
                        flag = False
            else:
                clf.fit(np.asarray(points), np.array(labels))
                cur_point = compound.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str[1: (len(cur_str) - 1)] not in used:
                    # decide if ask oracle for help
                    i += 1
                    prob = clf.predict_proba(cur_point)
                    if 0.1 <= prob[0][0] <= 0.9:
                        points.append(cur_point)
                        cur_label = oracle.oracle2(cur_point, features)
                        labels.append(cur_label)
                        query_count += 1
                        used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                        clf.fit(np.asarray(points), np.array(labels))
                        pred = clf.predict(testset)
                        cur_acc = error.test_error(pred, true_labels)
                        print cur_acc, " ", query_count, " ", cur_label, " ", prob[0][0], " ", prob[0][1]
                        accuracy.append(cur_acc)
    else:
        # random learner
        while i < budget:
            cur_point = compound.next_compound(features)
            cur_str = np.array_str(np.char.mod('%d', cur_point))
            if cur_str[1: (len(cur_str) - 1)] not in used:
                points.append(cur_point)
                cur_label = oracle.oracle2(cur_point, features)
                if cur_label == 1:
                    flag = False
                labels.append(cur_label)
                used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                query_count += 1
                i += 1
                if not flag:
                    clf.fit(np.asarray(points), np.array(labels))
                    pred = clf.predict(testset)
                    cur_acc = error.test_error(pred, true_labels)
                    print cur_acc, " ", query_count, " ", cur_label
                    accuracy.append(cur_acc)
    plt.plot(accuracy)
    plt.show()
    print "f1", f1_score(pred, true_labels[0:250])
    return


def svm_learner(budget):
    accuracy = []
    data = csv_reader('resources/pool.csv')
    testset = csv_reader('resources/testSet.csv')
    true_labels = oracle.read_mat()
    used = {}

    # do nothing about model until reasonable training subset achieved
    [row, col] = data.shape
    preds = np.zeros(row)
    selected = []
    labels = []
    query = 0
    # query each point until get one with label 1
    while 1:
        r = compound.next_compound(data)
        r_str = np.array_str(np.char.mod('%d', r))
        if r_str[1: (len(r_str) - 1)] not in used:
            r_label = oracle.oracle2(r, data)
            query += 1
            used[r_str[1: (len(r_str) - 1)]] = r_label
            selected.append(r.tolist())
            labels.append(r_label)
            accuracy.append(error.generalization_error(preds, true_labels))
            if np.sum(labels) == 1 and len(labels) > 1:
                accuracy.pop()
                break
    x = np.array(selected)
    y = np.array(labels)
    clf = SVC(kernel='linear')
    clf.fit(x, y)
    preds = clf.predict(data)
    accuracy.append(error.generalization_error(preds, true_labels))

    num = 2543 - len(used)
    i = 0
    while i < num and query < budget:
        r = compound.next_compound(data)
        r_str = np.array_str(np.char.mod('%d', r))
        if r_str[1: (len(r_str) - 1)] not in used:
            i += 1
            distance = clf.decision_function(r)
            if np.abs(distance[0]) <= 0.78:
                x = np.vstack([x, r])
                r_label = oracle.oracle2(r, data)
                y = np.hstack([y.tolist(), r_label])
                query += 1
                clf.fit(x, y)
                preds = clf.predict(testset)
                accuracy.append(error.test_error(preds, true_labels))
    plt.plot(accuracy)
    plt.show()
    print f1_score(preds, true_labels[0:250])
    return


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
    # stream_learner("lr", "select", 256)
    svm_learner(256)
