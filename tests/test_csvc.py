# -*- coding: utf-8 -*-

import unittest
import numpy as np
from sklearn import svm
from mdsvm import csvc


class TestCsvc(unittest.TestCase):

    def test_objval_poly(self):
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
        max_iter = 500000
        C = 1e1
        gamma = 0.01
        tol = 1e-7
        kernel = 'poly'
        # Training
        model_md = csvc.SVC(C=C, kernel=kernel, max_iter=max_iter, tol=tol, gamma=gamma)
        model_sk = svm.SVC(C=C, kernel=kernel, max_iter=max_iter, tol=tol, gamma=gamma)
        model_md.fit(x, y)
        model_sk.fit(x, y)
        # Compute objective value
        obj_sk = np.dot(
            model_sk.dual_coef_[0],
            np.dot(
                model_md._kernel_function(x[model_sk.support_], x[model_sk.support_]),
                model_sk.dual_coef_[0]
            )
        ) / 2 - sum(model_sk.dual_coef_[0] * y[model_sk.support_])
        print 'obj_sklearn:', obj_sk
        obj_md = np.dot(
            model_md._alpha*y,
            np.dot(model_md._kernel_function(x, x), model_md._alpha * y)
        ) / 2 - sum(model_md._alpha)
        print 'obj_mdsvm:', obj_md
        # Check objective values of two models
        np.testing.assert_almost_equal(obj_md, obj_sk, 5)
