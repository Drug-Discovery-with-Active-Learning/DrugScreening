# author = 'yanhe'


import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import get_compound as gc
import get_oracle as go
import get_error as ge


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


def rfc_learner(option):
    features = csv_reader('resources/pool.csv')
    testset = csv_reader('resources/testSet.csv')
    true_labels = go.read_mat()
    accuracy = []
    points = []
    labels = []
    used = {}
    flag = True
    query_count = 0
    i = 0
    if option == "select":
        while i < 2543 and query_count < 500:
            if flag:
                # call next compound until get one point with label 1
                cur_point = gc.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str[1: (len(cur_str) - 1)] not in used:
                    points.append(cur_point)
                    cur_label = go.oracle2(cur_point, features)
                    labels.append(cur_label)
                    used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                    query_count += 1
                    i += 1
                    if cur_label == 1:
                        flag = False
            else:
                clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
                clf.fit(np.asarray(points), np.array(labels))
                cur_point = gc.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str[1: (len(cur_str) - 1)] not in used:
                    # decide if ask oracle for help
                    prob = clf.predict_proba(cur_point)
                    if 0.4 < prob[0][0] < 0.6:
                        points.append(cur_point)
                        cur_label = go.oracle2(cur_point, features)
                        labels.append(cur_label)
                        query_count += 1
                        i += 1
                        used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                        clf.fit(np.asarray(points), np.array(labels))
                        pred = clf.predict(testset)
                        cur_acc = ge.test_error(pred, true_labels)
                        print cur_acc, " ", query_count, " ", cur_label, " ", prob[0][0], " ", prob[0][1]
                        accuracy.append(cur_acc)
    else:
        while i < 500:
            cur_point = gc.next_compound(features)
            cur_str = np.array_str(np.char.mod('%d', cur_point))
            if cur_str[1: (len(cur_str) - 1)] not in used:
                points.append(cur_point)
                cur_label = go.oracle2(cur_point, features)
                if cur_label == 1:
                    flag = False
                labels.append(cur_label)
                used[cur_str[1: (len(cur_str) - 1)]] = cur_label
                query_count += 1
                i += 1
                if not flag:
                    clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
                    clf.fit(np.asarray(points), np.array(labels))
                    pred = clf.predict(testset)
                    cur_acc = ge.test_error(pred, true_labels)
                    print cur_acc, " ", query_count, " ", cur_label
                    accuracy.append(cur_acc)
    plt.plot(accuracy)
    plt.show()
    return

if __name__ == "__main__":
    rfc_learner("select")
