
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

import cv2
from tensorflow.contrib.learn.python.learn.datasets.base import Dataset

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def black_white(n_samples=100, noise=None, seed=None, factor=0.8, n_classes=2, *args, **kwargs):
    X1 = np.ones([n_samples // 2, 28 * 28], np.uint8) * 255
    Y1 = ([[1] + [0] * 9]) * (n_samples // 2)
    X2 = np.zeros([n_samples - n_samples // 2, 28 * 28], np.uint8)
    Y2 = ([[0] + [1] + [0] * 8]) * (n_samples - n_samples // 2)
    indices = np.random.permutation(range(n_samples))
    X = np.concatenate((X1, X2))
    Y = np.concatenate((Y1, Y2))
    #import pdb;pdb.set_trace();
    return Dataset(data=X[indices], target=Y[indices])


def rect():
    img = np.zeros((28,28,3), np.uint8)
    cv2.rectangle(img, (10, 10), (20, 20), (255, 255, 255), -1)
    return img


def circle():
    img = np.zeros((28,28,3), np.uint8)
    cv2.circle(img, (14, 14), 8, (255, 255, 255), -1)
    return img


def circlesAndRects(n_samples=100):
    klass_reps = [rect(), circle()]
    X = []
    Y = []
    length = n_samples // len(klass_reps)
    for i, klass in enumerate(klass_reps):
        X.extend([klass] * length)
        y = np.zeros(10)
        y[i] = 1
        Y.extend([y] * length)
    X = np.array(X)
    Y = np.array(Y)
    indices = np.random.permutation(range(n_samples))
    #import pdb;pdb.set_trace();
    return Dataset(data=X[indices], target=Y[indices])



