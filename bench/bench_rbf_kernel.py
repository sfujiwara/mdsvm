# -*- coding: utf-8 -*-

from mdsvm import kernel_functions
import sklearn.metrics
import numpy as np
import time


np.random.seed(0)
num_p = 1500
num_n = 1500
dim = 100
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
x1 = np.vstack([x_p, x_n])
x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
x2 = np.vstack([x_p, x_n])

gamma = 0.5

rbf_kernel_mdsvm = kernel_functions.get_rbf_kernel(gamma=gamma)

t = time.time()
for _ in range(1):
    sklearn.metrics.pairwise.rbf_kernel(x1, x2, gamma=gamma)
print 'scikit-learn: {} sec'.format(time.time() - t)

t = time.time()
for _ in range(1):
    rbf_kernel_mdsvm(x1, x2)
print 'MDSVM: {} sec'.format(time.time() - t)
