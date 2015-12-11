# author = 'yanhe'


import csv
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import get_compound as gc
import get_oracle as go


def csv_reader():
    feature = []
    with open('resources/testSet.csv', 'r') as pool_file:
        file_reader = csv.reader(pool_file)
        for row in file_reader:
            cur = np.array(row)
            num = map(int, cur)
            feature.append(num)
    data = np.array(feature)
    return data


def rfc_learner():
    accuracy = []
    points = []
    labels = []
    used = set()
    flag = True
    query_count = 0
    for i in xrange(0, 2543):
        if flag:
            # call next compound until get one point with label 1
            query_count += 1
            cur_point = gc.next_compound()
            if cur_point not in used:
                used.add(cur_point)
                points.append(cur_point)
                cur_label = go.oracle2(cur_point)
                labels.append(cur_label)
                print cur_label
                if cur_label == 1:
                    flag = False
        else:
            clf = RandomForestClassifier(n_estimators=10, criterion='entropy')
            clf.fit(np.asarray(points), np.array(labels))
            cur_point = gc.next_compound()
            if cur_point not in used:
                used.add(cur_point)
                # decide if ask oracle for help




if __name__ == "__main__":
    rfc_learner()
