# __ author__ = 'yanhe'


import csv
import random
import numpy as np


def next_compound(features):
    size = len(features)
    pick = random.sample(range(size), 1)[0]
    return features[pick]
