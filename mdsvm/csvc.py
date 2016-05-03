# -*- coding: utf-8 -*-

import numpy as np
import kernel_functions as kf


class SVC:

    def __init__(self, C=1e0, kernel='linear', tol=1e-3, max_iter=1000,
                 gamma=1e0, degree=3, coef0=0):
        self._EPS = 1e-5
        self._TAU = 1e-12
        self._cache = {}
        self.tol = tol
        self.max_iter = max_iter
        self.C = C
        self.gamma = gamma
        self.degree, self.coef0 = degree, coef0
        self.kernel = kernel
        self._kernel_function = None
        self._alpha = None
        self.intercept_ = None
        self._grad = None
        self.itr = None
        self._ind_y_p, self._ind_y_n = None, None
        self._i_low, self._i_up = None, None
        self.set_kernel_function(kernel)

    def _init_solution(self, y):
        num = len(y)
        self._i_low = y < 0
        self._i_up = y > 0
        # Initialize the dual coefficients and gradient
        self._alpha = np.zeros(num)
        self._grad = - np.ones(num)
        self._cache = {}

    def set_kernel_function(self, kernel):
        if kernel == 'linear':
            self._kernel_function = kf.linear_kernel
        elif kernel == 'rbf' or kernel == 'gaussian':
            self._kernel_function = kf.get_rbf_kernel(self.gamma)
        elif kernel == 'polynomial' or kernel == 'poly':
            self._kernel_function = kf.get_polynomial_kernel(self.degree, self.coef0, self.gamma)
        else:
            raise ValueError('{} is undefined name as kernel function'.format(kernel))

    def _get_q(self, x, y, ind):
        if ind in self._cache:
            qi = self._cache[ind]
        else:
            qi = self._kernel_function(x, x[ind]).ravel()
            qi *= y
            if y[ind] < 0:
                qi *= y[ind]
            self._cache[ind] = qi
            # qi = self._cache[ind] = self._kernel_function(x, x[ind]).ravel() * y * y[ind]
        return qi

    def _select_working_set1(self, y):
        ygrad = y * self._grad
        # Convert boolean mask to index
        i_up = self._i_up.nonzero()[0]
        i_low = self._i_low.nonzero()[0]
        ind_ws1 = i_up[np.argmin(ygrad[i_up])]
        ind_ws2 = i_low[np.argmax(ygrad[i_low])]
        return ind_ws1, ind_ws2

    def fit(self, x, y):
        self._init_solution(y)
        num, _ = x.shape
        # Start the iterations of SMO algorithm
        for itr in xrange(self.max_iter):
            # Select two indices of variables as working set
            ind_ws1, ind_ws2 = self._select_working_set1(y)
            # Check stopping criteria: m(a_k) <= M(a_k) + tolerance
            m_lb = - y[ind_ws1] * self._grad[ind_ws1]
            m_ub = - y[ind_ws2] * self._grad[ind_ws2]
            kkt_violation = m_lb - m_ub
            # print 'KKT Violation:', kkt_violation
            if kkt_violation <= self.tol:
                print 'Converged!', 'Iter:', itr, 'KKT Violation:', kkt_violation
                break
            # Compute (or get from cache) two columns of gram matrix
            qi = self._get_q(x, y, ind_ws1)
            qj = self._get_q(x, y, ind_ws2)
            # Construct sub-problem
            qii, qjj, qij = qi[ind_ws1], qj[ind_ws2], qi[ind_ws2]
            # Solve sub-problem
            if y[ind_ws1] * y[ind_ws2] > 0:  # The case where y_i equals y_j
                v2 = -1.
                d_max = min(self.C - self._alpha[ind_ws1], self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], self._alpha[ind_ws2] - self.C)
            else:  # The case where y_i equals y_j
                v2 = 1.
                d_max = min(self.C - self._alpha[ind_ws1], self.C - self._alpha[ind_ws2])
                d_min = max(-self._alpha[ind_ws1], -self._alpha[ind_ws2])
            quad_coef = qii + qjj + 2 * v2 * qij
            quad_coef = max(quad_coef, self._TAU)
            d = - (self._grad[ind_ws1] + self._grad[ind_ws2] * v2) / quad_coef
            d = max(min(d, d_max), d_min)
            # Update dual coefficients
            self._alpha[ind_ws1] += d
            self._alpha[ind_ws2] += d * v2
            # Update the gradient
            # self._grad += d * qi + d * v2 * qj
            self._grad += d * qi
            self._grad += d * v2 * qj
            # if v2 > 0:
            #     self._grad += d * qj
            # else:
            #     self._grad -= d * qj
            # Update I_up with respect to ind_ws1 and ind_ws2
            self._update_iup_and_ilow(y, ind_ws1)
            self._update_iup_and_ilow(y, ind_ws2)
        else:
            print 'Exceed maximum iteration'
            print 'KKT Violation:', kkt_violation
        # Set results after optimization procedure
        self._set_result(x, y)
        self.intercept_ = (m_lb + m_ub) / 2.
        self.itr = itr + 1

    def _update_iup_and_ilow(self, y, ind):
        # Update I_up with respect to ind
        if (y[ind] > 0) and (self._alpha[ind] / self.C <= 1 - self._EPS):
            self._i_up[ind] = True
        elif (y[ind] < 0) and (self._EPS <= self._alpha[ind] / self.C):
            self._i_up[ind] = True
        else:
            self._i_up[ind] = False
        # Update I_low with respect to ind
        if (y[ind] > 0) and (self._EPS <= self._alpha[ind] / self.C):
            self._i_low[ind] = True
        elif (y[ind] < 0) and (self._alpha[ind] / self.C <= 1 - self._EPS):
            self._i_low[ind] = True
        else:
            self._i_low[ind] = False

    def _set_result(self, x, y):
        self.support_ = np.where(self._EPS < (self._alpha / max(self._alpha)))[0]
        self.support_vectors_ = x[self.support_]
        self.dual_coef_ = self._alpha[self.support_] * y[self.support_]
        # Compute w when using linear kernel
        if self._kernel_function == kf.linear_kernel:
            self.coef_ = np.sum(self.dual_coef_ * x[self.support_].T, axis=1)

    def decision_function(self, x):
        return np.sum(self._kernel_function(x, self.support_vectors_) * self.dual_coef_, axis=1) + self.intercept_

    def predict(self, x):
        return np.sign(self.decision_function(x))

    def score(self, x, y):
        return sum(self.decision_function(x) * y > 0) / float(len(y))

    def to_dict(self):
        result = {
            'type': 'CSVC',
            'dualCoef': self.dual_coef_.tolist(),
            'intercept': self.intercept_,
            'supportVectors': self.support_vectors_.tolist(),
            'kernel': self.kernel,
            'tol': self.tol,
            'hyperParameters': {
                'C': self.C,
                'gamma': self.gamma,
                'coef0': self.coef0,
                'degree': self.degree
            }
        }
        return result
