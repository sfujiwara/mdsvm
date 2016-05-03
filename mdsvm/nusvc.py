# -*- coding: utf-8 -*-

import numpy as np
import kernel_functions as kf

# TODO: b, rho の求め方を修正

class NuSVC:

    def __init__(self, nu=0.5, kernel=kf.linear_kernel, tol=1e-3, max_iter=1000,
                 gamma=1e0, degree=3, coef0=0):
        self._EPS = 1e-5
        self._TAU = 1e-12
        self._cache = {}
        self.tol = tol
        self.max_iter = max_iter
        self.nu = nu
        self.gamma = gamma
        self.degree, self.coef0 = degree, coef0
        self.kernel = None
        self._alpha = None
        self.intercept_ = None
        self._grad = None
        self.itr = None
        self._ind_y_p, self._ind_y_n = None, None
        self._i_low, self._i_up = None, None
        self.set_kernel_function(kernel)

    def _init_solution(self, y):
        num = len(y)
        self._i_low = np.ones(num, dtype=bool)
        self._i_up = np.ones(num, dtype=bool)
        # Initialize cache of kernel gram matrix
        self._cache = {}
        # Initialize the dual coefficients and gradient
        self._alpha = np.zeros(num)
        self._alpha[y > 0] = num * self.nu / sum(y > 0) / 2
        self._alpha[y < 0] = num * self.nu / sum(y < 0) / 2
        # self._grad = np.dot(self.kernel((x.T*y).T, (x.T*y).T), self._alpha)

    def _init_gradient(self, x, y):
        num = len(y)
        self._grad = np.zeros(num)
        for i in range(num):
            qi = self.kernel(x, x[i]).ravel() * y * y[i]
            self._cache[i] = qi
            self._grad[i] = np.dot(qi, self._alpha)

    def set_kernel_function(self, kernel):
        if callable(kernel):
            self.kernel = kernel
        elif kernel == 'linear':
            self.kernel = kf.linear_kernel
        elif kernel == 'rbf' or kernel == 'gaussian':
            print 'hello'
            self.kernel = kf.get_rbf_kernel(self.gamma)
        elif kernel == 'polynomial' or kernel == 'poly':
            self.kernel = kf.get_polynomial_kernel(self.degree, self.coef0, self.gamma)
        else:
            raise ValueError('{} is undefined name as kernel function'.format(kernel))

    def _select_working_set1(self, y):
        minus_y_times_grad = - y * self._grad
        # Select working set from positive class
        i_up_p = (self._i_up * (y > 0)).nonzero()[0]
        i_low_p = (self._i_low * (y > 0)).nonzero()[0]
        ind_ws1_p = i_up_p[np.argmax(minus_y_times_grad[i_up_p])]
        ind_ws2_p = i_low_p[np.argmin(minus_y_times_grad[i_low_p])]
        # Select working set from negative class
        i_up_n = (self._i_up * (y < 0)).nonzero()[0]
        i_low_n = (self._i_low * (y < 0)).nonzero()[0]
        ind_ws1_n = i_up_n[np.argmax(minus_y_times_grad[i_up_n])]
        ind_ws2_n = i_low_n[np.argmin(minus_y_times_grad[i_low_n])]
        return ind_ws1_p, ind_ws2_p, ind_ws1_n, ind_ws2_n

    def fit(self, x, y):
        self._init_solution(y)
        self._init_gradient(x, y)
        num, _ = x.shape
        # Start the iterations of SMO algorithm
        for itr in xrange(self.max_iter):
            print itr
            # Select two indices of variables as working set
            ind_ws1_p, ind_ws2_p, ind_ws1_n, ind_ws2_n = self._select_working_set1(y)
            # Check stopping criteria: m(a_k) <= M(a_k) + tolerance
            m_lb_p = - y[ind_ws1_p] * self._grad[ind_ws1_p]
            m_ub_p = - y[ind_ws2_p] * self._grad[ind_ws2_p]
            m_lb_n = - y[ind_ws1_n] * self._grad[ind_ws1_n]
            m_ub_n = - y[ind_ws2_n] * self._grad[ind_ws2_n]
            kkt_violation_p = m_lb_p - m_ub_p
            kkt_violation_n = m_lb_n - m_ub_n
            if kkt_violation_p > kkt_violation_n:
                ind_ws1, ind_ws2 = ind_ws1_p, ind_ws2_p
                kkt_violation = kkt_violation_p
            else:
                ind_ws1, ind_ws2 = ind_ws1_n, ind_ws2_n
                kkt_violation = kkt_violation_n
            # print 'KKT Violation:', kkt_violation
            if kkt_violation <= self.tol:
                print 'Converged!', 'Iter:', itr, 'KKT Violation:', kkt_violation
                break
            # Compute (or get from cache) two columns of gram matrix
            if ind_ws1 in self._cache:
                qi = self._cache[ind_ws1]
            else:
                qi = self.kernel(x, x[ind_ws1]).ravel() * y * y[ind_ws1]
                self._cache[ind_ws1] = qi
            if ind_ws2 in self._cache:
                qj = self._cache[ind_ws2]
            else:
                qj = self.kernel(x, x[ind_ws2]).ravel() * y * y[ind_ws2]
                self._cache[ind_ws2] = qj
            # Construct sub-problem
            qii, qjj, qij = qi[ind_ws1], qj[ind_ws2], qi[ind_ws2]
            # Solve sub-problem
            if y[ind_ws1] * y[ind_ws2] > 0:  # The case where y_i equals y_j
                v1, v2 = 1., -1.
                d_max = min(1 - self._alpha[ind_ws1], self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], self._alpha[ind_ws2] - 1)
            else:  # The case where y_i equals y_j
                v1, v2 = 1., 1.
                d_max = min(self.C - self._alpha[ind_ws1], self.C - self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], -self._alpha[ind_ws2])
            quad_coef = v1**2 * qii + v2**2 * qjj + 2 * v1 * v2 * qij
            quad_coef = max(quad_coef, self._TAU)
            d = - (self._grad[ind_ws1] * v1 + self._grad[ind_ws2] * v2) / quad_coef
            d = max(min(d, d_max), d_min)
            # Update dual coefficients
            self._alpha[ind_ws1] += d * v1
            self._alpha[ind_ws2] += d * v2
            # Update the gradient
            self._grad += d * v1 * qi + d * v2 * qj
            # Update I_up with respect to ind_ws1 and ind_ws2
            self._update_iup_and_ilow(y, ind_ws1)
            self._update_iup_and_ilow(y, ind_ws2)
        else:
            print 'Exceed maximum iteration'
            print 'KKT Violation:', kkt_violation
        # Set results after optimization procedure
        self._set_result(x, y)
        self.intercept_ = ((m_lb_p + m_ub_p) + (m_lb_n + m_ub_n)) / 4.
        self.itr = itr + 1

    def _update_iup_and_ilow(self, y, ind):
        # Update I_up with respect to ind
        if (y[ind] > 0) and (self._alpha[ind] <= 1 - self._EPS):
            self._i_up[ind] = True
        elif (y[ind] < 0) and (self._EPS <= self._alpha[ind]):
            self._i_up[ind] = True
        else:
            self._i_up[ind] = False
        # Update I_low with respect to ind
        if (y[ind] > 0) and (self._EPS <= self._alpha[ind]):
            self._i_low[ind] = True
        elif (y[ind] < 0) and (self._alpha[ind] <= 1 - self._EPS):
            self._i_low[ind] = True
        else:
            self._i_low[ind] = False

    def _set_result(self, x, y):
        self.support_ = np.where(self._EPS < (self._alpha / max(self._alpha)))[0]
        self.support_vectors_ = x[self.support_]
        self.dual_coef_ = self._alpha[self.support_] * y[self.support_]
        # Compute w when using linear kernel
        if self.kernel == kf.linear_kernel:
            self.coef_ = np.sum(self.dual_coef_ * x[self.support_].T, axis=1)

    def decision_function(self, x):
        return np.sum(self.kernel(x, self.support_vectors_) * self.dual_coef_, axis=1) + self.intercept_

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        return sum(self.decision_function(x) * y > 0) / float(len(y))


if __name__ == '__main__':
    # Create toy problem
    np.random.seed(0)
    num_p = 250
    num_n = 150
    dim = 8
    x_p = np.random.multivariate_normal(np.ones(dim) * 1, np.eye(dim), num_p)
    x_n = np.random.multivariate_normal(np.ones(dim) * 2, np.eye(dim), num_n)
    x = np.vstack([x_p, x_n])
    y = np.array([1.] * num_p + [-1.] * num_n)
    # Model
    model = NuSVC()
    model.fit(x, y)
    print model.score(x, y)
