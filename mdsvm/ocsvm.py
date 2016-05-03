# -*- coding: utf-8 -*-

import warnings
import numpy as np
import kernel_functions as kf


class OneClassSVM:

    def __init__(self, nu=0.5, kernel='rbf', tol=1e-3, max_iter=1000, gamma=1e0, degree=3, coef0=0):
        self._EPS = 1e-5
        self._TAU = 1e-12
        self._cache = {}
        self.tol = tol
        self.max_iter = max_iter
        self.nu = nu
        self.gamma = gamma
        self.degree, self.coef0 = degree, coef0
        self._kernel_function = None
        self.kernel = kernel
        self._alpha = None
        self.intercept_ = None
        self._grad = None
        self.itr = None
        self.set_kernel_function(kernel)

    def _init_solution(self, x):
        num, _ = x.shape
        self._i_alf_gt_zero = np.ones(num, dtype=bool)
        self._i_alf_lt_one = np.ones(num, dtype=bool)
        # Initialize cache of kernel gram matrix
        self._cache = {}
        # Initialize the dual coefficients
        self._alpha = np.zeros(num)
        self._alpha[:int(self.nu * num)] = 1.
        self._alpha[int(self.nu * num)] = self.nu * num - int(self.nu * num)
        # Initialize gradient
        self._grad = np.zeros(num)
        # self._grad = np.dot(self.kernel(x, x), self._alpha)
        for i in xrange(int(self.nu * num) + 1):
            self._grad += self._get_q(x, i) * self._alpha[i]

    def set_kernel_function(self, kernel):
        if kernel == 'linear':
            self._kernel_function = kf.linear_kernel
        elif kernel == 'rbf':
            self._kernel_function = kf.get_rbf_kernel(self.gamma)
        elif kernel == 'poly':
            self._kernel_function = kf.get_polynomial_kernel(self.degree, self.coef0, self.gamma)
        else:
            raise ValueError('{} is undefined name as kernel function'.format(kernel))

    def _select_working_set(self):
        ind_ws1 = self._i_alf_gt_zero.nonzero()[0][np.argmax(self._grad[self._i_alf_gt_zero])]
        ind_ws2 = self._i_alf_lt_one.nonzero()[0][np.argmin(self._grad[self._i_alf_lt_one])]
        return ind_ws1, ind_ws2

    def _get_q(self, x, ind):
        if ind in self._cache:
            qi = self._cache[ind]
        else:
            qi = self._cache[ind] = self._kernel_function(x, x[ind]).ravel()
        return qi

    def fit(self, x):
        num, _ = x.shape
        self._init_solution(x)
        # self._init_gradient(x)
        # Start the iterations of SMO algorithm
        for itr in xrange(self.max_iter):
            # Select two indices of variables as working set
            ind_ws1, ind_ws2 = self._select_working_set()
            # Check stopping criteria: m(a_k) <= M(a_k) + tolerance
            kkt_violation = self._grad[ind_ws1] - self._grad[ind_ws2]
            # print 'KKT Violation:', kkt_violation
            if kkt_violation <= self.tol:
                # print 'Converged!', 'Iter:', itr, 'KKT Violation:', kkt_violation
                break
            # Compute (or get from cache) two columns of gram matrix
            qi = self._get_q(x, ind_ws1)
            qj = self._get_q(x, ind_ws2)
            # Construct sub-problem
            qii, qjj, qij = qi[ind_ws1], qj[ind_ws2], qi[ind_ws2]
            # Solve sub-problem
            d_max = min(1 - self._alpha[ind_ws1], self._alpha[ind_ws2])
            d_min = max(-self._alpha[ind_ws1], self._alpha[ind_ws2] - 1)
            quad_coef = qii + qjj - 2 * qij
            quad_coef = max(quad_coef, self._TAU)
            d = - (self._grad[ind_ws1] - self._grad[ind_ws2]) / quad_coef
            d = max(min(d, d_max), d_min)
            # Update dual coefficients
            self._alpha[ind_ws1] += d
            self._alpha[ind_ws2] -= d
            # Update the gradient
            self._grad += d * qi - d * qj
            # Update I_up with respect to ind_ws1 and ind_ws2
            self._i_alf_gt_zero[ind_ws1] = self._EPS <= self._alpha[ind_ws1]
            self._i_alf_gt_zero[ind_ws2] = self._EPS <= self._alpha[ind_ws2]
            self._i_alf_lt_one[ind_ws1] = self._alpha[ind_ws1] <= (1 - self._EPS)
            self._i_alf_lt_one[ind_ws2] = self._alpha[ind_ws2] <= (1 - self._EPS)
        else:
            print 'Exceed maximum iteration'
            print 'KKT Violation:', kkt_violation
            warnings.warn("Exceed maximum iteration")

        # Set results after optimization procedure
        self._set_result(x)
        self.intercept_ = -(self._grad[ind_ws1] + self._grad[ind_ws2]) / 2.
        self.itr = itr + 1

    def _set_result(self, x):
        self.support_ = np.where(self._EPS < (self._alpha / max(self._alpha)))[0]
        self.support_vectors_ = x[self.support_]
        self.dual_coef_ = self._alpha[self.support_]
        # Compute w when using linear kernel
        if self._kernel_function == kf.linear_kernel:
            self.coef_ = np.sum(self.dual_coef_ * x[self.support_].T, axis=1)

    def decision_function(self, x):
        return np.sum(self._kernel_function(x, self.support_vectors_) * self.dual_coef_, axis=1) + self.intercept_

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def to_dict(self):
        result = {
            'type': 'OneClassSVM',
            'dual_coef_': self.dual_coef_.tolist(),
            'intercept_': self.intercept_,
            'support_vectors_': self.support_vectors_.tolist(),
            'kernel': self.kernel,
            'tol': self.tol,
            'hyperParameters': {
                'nu': self.nu,
                'gamma': self.gamma,
                'coef0': self.coef0,
                'degree': self.degree
            }
        }
        return result

if __name__ == '__main__':
    from sklearn import svm
    import cProfile
    # Create toy problem
    np.random.seed(0)
    num = 1000
    dim = 2
    x = np.vstack([
        np.random.multivariate_normal(np.ones(dim) * 3, np.eye(dim), 10),
        np.random.multivariate_normal(np.ones(dim), np.array([[1, -0.8], [-0.8, 1]]), num)
    ])
    # Model
    nu = 0.1
    tol = 1e-3
    gamma = 0.1
    model = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma, max_iter=10000, tol=tol)
    model.fit(x)
    clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma, max_iter=10000, tol=tol, shrinking=False)
    clf.fit(x)
    # print model.dual_coef_
    # print clf.dual_coef_
    cProfile.run('model.fit(x)')
    # print model.score(x)

    import matplotlib.pyplot as plt
    is_outlier = model.decision_function(x) < -10
    # is_outlier = clf.decision_function(x).flatten() < -0
    plt.plot(x[:, 0], x[:, 1], 'x')
    plt.plot(x[is_outlier, 0], x[is_outlier, 1], 'rs')
    plt.grid()
    plt.show()
