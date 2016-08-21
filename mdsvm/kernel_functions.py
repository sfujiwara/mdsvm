# -*- coding: utf-8 -*-

import numpy as np


def linear_kernel(x1, x2):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    return np.dot(x1, x2.T)


# k(x, y) = exp(- gamma || x1 - x2 ||^2)
def get_rbf_kernel(gamma):
    def rbf_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        x1x1 = (x1 * x1).sum(axis=1)[:, np.newaxis]
        x2x2 = (x2 * x2).sum(axis=1)[np.newaxis, :]
        kmat = np.dot(x1, x2.T)
        kmat *= -2
        kmat += x1x1
        kmat += x2x2
        np.maximum(kmat, 0, out=kmat)
        kmat *= -gamma
        np.exp(kmat, kmat)
        return kmat
    return rbf_kernel


# k(x1, x2) = (gamma<x1, x2> + coef0)^degree
def get_polynomial_kernel(degree, coef0, gamma):
    def polynomial_kernel(x1, x2):
        x1 = np.atleast_2d(x1)
        x2 = np.atleast_2d(x2)
        kmat = np.dot(x1, x2.T)
        kmat *= gamma
        kmat += coef0
        kmat **= degree
        return kmat
    return polynomial_kernel
