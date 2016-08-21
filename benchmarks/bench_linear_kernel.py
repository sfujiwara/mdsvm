# -*- coding: utf-8 -*-

from mdsvm import kernel_functions
import sklearn.metrics
import numpy as np
import time


np.random.seed(0)
num_p = 1500
num_n = 1500
dim = 50
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
x = np.vstack([x_p, x_n])

t = time.time()
for i in range(50):
    kernel_functions.linear_kernel(x, x)
print 'MDSVM: {} sec'.format(time.time() - t)

t = time.time()
for i in range(50):
    sklearn.metrics.pairwise.linear_kernel(x, x)
print 'scikit-learn: {} sec'.format(time.time() - t)
