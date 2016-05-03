# -*- coding: utf-8 -*-

from mdsvm import csvc
from sklearn import svm
import numpy as np
import time


# Data set
np.random.seed(0)
num_p = 1500
num_n = 1500
dim = 300
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 1.5, np.eye(dim), num_n)
x = np.vstack([x_p, x_n])
y = np.array([1.] * num_p + [-1.] * num_n)

# Hyper parameters
cost = 1e0
max_iter = 10000000

clf_mdsvm = csvc.SVC(C=cost, kernel='poly', max_iter=max_iter)
clf_sklearn = svm.SVC(C=cost, kernel='poly', max_iter=max_iter, shrinking=False, gamma=1.)

t = time.time()
clf_mdsvm.fit(x, y)
print 'MDSVM: {} sec'.format(time.time() - t)

t = time.time()
clf_sklearn.fit(x, y)
print 'scikit-learn: {} sec'.format(time.time() - t)
