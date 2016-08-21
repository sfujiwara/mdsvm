# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sklearn.metrics.pairwise
from mdsvm import kernel_functions


class TestKernelFunctions(unittest.TestCase):

    def test_rbf_kernel(self):
        # Create toy problem
        np.random.seed(0)
        num_p = 15
        num_n = 15
        dim = 20
        x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
        x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
        x = np.vstack([x_p, x_n])
        y = np.array([1.] * num_p + [-1.] * num_n)

        # Set parameters
        gamma = 0.01

        # Compute kernel gram matrix
        rbf_kernel_mdsvm = kernel_functions.get_rbf_kernel(gamma)
        kmat_mdsvm = rbf_kernel_mdsvm(x, x[0])
        kmat_sklearn = sklearn.metrics.pairwise.rbf_kernel(x, x[0], gamma)
        np.testing.assert_almost_equal(kmat_mdsvm, kmat_sklearn, 5)

        # Compute kernel gram matrix
        linear_kernel_mdsvm = kernel_functions.linear_kernel
        kmat_mdsvm = linear_kernel_mdsvm(x, x[0])
        kmat_sklearn = sklearn.metrics.pairwise.linear_kernel(x, x[0])
        np.testing.assert_almost_equal(kmat_mdsvm, kmat_sklearn, 5)
