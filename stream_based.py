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


def rfc_learner():
    features = csv_reader('resources/pool.csv')
    testset = csv_reader('resources/testSet.csv')
    true_labels = go.read_mat()
    accuracy = []
    points = []
    labels = []
    used = set()
    flag = True
    query_count = 0
    for i in xrange(0, 2543):
        while query_count < 256:
            if flag:
                # call next compound until get one point with label 1
                cur_point = gc.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str not in used:
                    used.add(cur_str[1: (len(cur_str) - 1)])
                    points.append(cur_point)
                    cur_label = go.oracle2(cur_point, features)
                    labels.append(cur_label)
                    query_count += 1
                    if cur_label == 1:
                        flag = False
            else:
                clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
                clf.fit(np.asarray(points), np.array(labels))
                pred = clf.predict(testset)
                cur_acc = ge.test_error(pred, true_labels)
                print cur_acc
                accuracy.append(cur_acc)
                cur_point = gc.next_compound(features)
                cur_str = np.array_str(np.char.mod('%d', cur_point))
                if cur_str not in used:
                    # decide if ask oracle for help
                    prob = clf.predict_proba(cur_point)
                    if 0.4 <= prob[0][1] <= 0.6:
                        used.add(cur_str[1: (len(cur_str) - 1)])
                        points.append(cur_point)
                        cur_label = go.oracle2(cur_point, features)
                        labels.append(cur_label)
                        query_count += 1
    plt.plot(accuracy)
    plt.show()
    return

if __name__ == "__main__":
    rfc_learner()
