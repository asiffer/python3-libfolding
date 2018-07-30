#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see http://www.gnu.org/licenses/.

@author: asr
"""

from ctypes import CDLL, POINTER, byref
from ctypes import c_double, c_int, c_bool, c_void_p, c_size_t, c_long
from math import log, sqrt, exp
from scipy.optimize import brenth, bisect
from numpy import float64
import numpy as np

PVAL_A = 0.4785
PVAL_B = 0.1946
PVAL_C = 2.0287

ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=float64, ndim=1, flags='CONTIGUOUS')
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='CONTIGUOUS')

LIBFOLDING = CDLL('libfolding.so')

LIBFOLDING.sf_new.argtypes = [c_size_t]
LIBFOLDING.sf_new.restype = c_void_p

LIBFOLDING.sf_update.argtypes = [c_void_p, ND_POINTER_1, c_size_t, c_bool, c_bool]

LIBFOLDING.sf_is_initialized.argtypes = [c_void_p]
LIBFOLDING.sf_is_initialized.restypes = [c_bool]

LIBFOLDING.sf_folding_test.argtypes = [c_void_p,
                                       POINTER(c_bool),
                                       POINTER(c_double),
                                       POINTER(c_double)]

LIBFOLDING.sf_mean.argtypes = [c_void_p, ND_POINTER_1]

LIBFOLDING.sf_cov.argtypes = [c_void_p, ND_POINTER_2]

LIBFOLDING.sf_s2star.argtypes = [c_void_p, ND_POINTER_1]

LIBFOLDING.sf_dump.argtypes = [c_void_p, ND_POINTER_2]

# (double * array_ptr, size_t n, size_t d, bool * unimodal, double * pval, double * Phi, int * usec)
LIBFOLDING.batch_folding_test.argtypes = [ND_POINTER_2, c_size_t, c_size_t,
                                          POINTER(c_bool), POINTER(c_double),
                                          POINTER(c_double), POINTER(c_long)]


class StreamFolding(object):
    """A python class to perform the folding test of unimodality over streaming
    data. This is a wrapper to the original C++ library (using ctypes)

    Attributes
    ----------
    _sf_ptr: c_void_p
        A pointer to the C++ object StreamFolding

    _depth: int
        The size of the sliding window

    _dim: int
        the dimension of the observations

    Examples
    --------
    >>> depth = 500
    >>> N = 1000
    >>> sf = StreamFolding(depth)
    >>> U = np.random.multivariate_normal([0, 0], [[2., 1, ], [1., 2.]], N)
    >>> for i in range(N):
    ...     sf.update(U[i, :])

    >>> sf.mean()
    array([-0.04876098,  0.03932685])
    >>> sf.s2star()
    array([-0.02220509,  0.05162952])
    >>> sf.cov()
    array([[2.15169376, 1.09152982],
           [1.09152982, 1.83930915]])

    >>> u, p, phi = sf.folding_test()

    """

    def __init__(self, depth):
        """
        Parameters
        ----------

        depth: int
            The size of the sliding window (statistics are preformed on the last
            depth observations)
        """
        self._sf_ptr = LIBFOLDING.sf_new(depth)
        self._depth = depth
        self._dim = None

    def is_initialized(self):
        """
        Check if the object has been initialized (i.e. if the statistics has
        been computed at least once)
        """
        return bool(LIBFOLDING.sf_is_initialized(self._sf_ptr))

    def update(self, new_data):
        """
        Main method (update the internal state according to a new observation)

        Parameters
        ----------

        x: numpy.array
            the new observation (it must be a vector)

        Raises
        ------

        TypeError
            When the dimension of the new observation is incorrect
        """
        obs_dim = len(new_data)
        if self._dim is None:
            self._dim = obs_dim
            LIBFOLDING.sf_update(self._sf_ptr, new_data, self._dim, True, False)
        elif obs_dim == self._dim:
            LIBFOLDING.sf_update(self._sf_ptr, new_data, self._dim, True, False)
        else:
            mess = "The dimension of the observation is incorrect "
            mess += "(assume {}, got {})".format(self._dim, obs_dim)
            raise TypeError(mess)

    def mean(self):
        """
        Compute the mean of the stored data

        Returns
        -------
        mean_data: numpy.ndarray
            the mean vector of the current observations (shape (d,))
        """
        mean_data = np.ones(self._dim)
        LIBFOLDING.sf_mean(self._sf_ptr, mean_data)
        return mean_data

    def s2star(self):
        """
        Compute the pivot s2*

        Returns
        -------
        folding_pivot: numpy.ndarray
            the folding pivot (shape (d,))
        """
        folding_pivot = np.ones(self._dim)
        LIBFOLDING.sf_s2star(self._sf_ptr, folding_pivot)
        return folding_pivot

    def cov(self):
        """
        Compute the covariance of the stored data

        Returns
        -------
        cov_data: numpy.ndarray
            the covariance matrix (shape (d,d))
        """
        cov_data = np.ones((self._dim, self._dim))
        LIBFOLDING.sf_cov(self._sf_ptr, cov_data)
        return cov_data

    def dump(self):
        """
        Return a dump of the stored data as a d x n matrix where n is the
        current number of stored data (equal to the window size in the cruising
        regime)

        Notes
        -----
        WARNING: the order of the observations can have an offset
        (the underlying container is a circular vector)
        """
        data = np.ones((self._dim, self._depth))
        LIBFOLDING.sf_dump(self._sf_ptr, data)
        return data

    def folding_test(self):
        """
        Perform the folding test of unimodality within the current window

        Returns
        -------

        unimodal: bool
            The result of the test : unimodal/multimodal

        p_val: float
            The significance of the test (the closer to 0 the better). Generally,
            with consider the test as "significant" when p_val < 0.05

        Phi: float
            The folding statistics
        """
        unimodal = c_bool(True)
        p_val = c_double(0.)
        phi = c_double(0.)
        LIBFOLDING.sf_folding_test(self._sf_ptr, byref(unimodal), byref(p_val), byref(phi))
        return unimodal.value, p_val.value, phi.value


def decision_bound(p_value, n, d):
    """
    Compute the decision bound q according to the desired p-value. The test would
    be significant if |phi-1|>q.

    Parameters
    ----------

    p_value: float
        between 0 and 1 (this is the probability to be in the uniform case)
    n: int
        the number of observations
    d: int
        the dimension
    """
    return PVAL_A * (p_value - PVAL_B * log(1 - p_value)) * (PVAL_C + log(d)) / sqrt(n)


def p_value(phi, n, d, e=1e-16):
    """
    Compute the p-value of a test

    Parameters
    ----------

    phi: float
        the folding statistics
    n: int
        the number of observations
    d: int
        the dimension
    """
    try:
        obj_fun = lambda p: (abs(phi - 1.) - decision_bound(1 - p, n, d))
        p_val = brenth(obj_fun, 0., 1.)
    except:

        p_val = exp(-abs(phi - 1.) * sqrt(n) / (PVAL_C + log(d)))
    return p_val


def batch_folding_test(X):
    """
    Perform statically the folding test of unimodality (pure python)

    Parameters
    ----------

    X: numpy.ndarray
        a d by n matrix (n observations in dimension d)
    """
    try:
        n, p = X.shape
    except:
        X = X.reshape(1, len(X))
        n, p = X.shape

    if n > p:  # if lines are observations, we transpose it
        X = X.T
        dim = p
        n_obs = n
    else:
        dim = n
        n_obs = p

    X_square_norm = (X * X).sum(axis=0)  # |X|²
    mat_cov = np.matrix(np.cov(X))  # cov(X)
    trace = np.trace(mat_cov)  # Tr(cov(X))
    cov_norm = np.cov(X, X_square_norm)[:-1, -1].reshape(-1, 1)  # cov(X,|X|²)
    pivot = 0.5 * np.linalg.solve(mat_cov, cov_norm)  # 0.5 * cov(X)^{-1} * cov(X,|X|²)
    X_reduced = np.sqrt(np.power(X - pivot, 2).sum(axis=0))  # |X-s*|²
    phi = pow(1. + dim, 2) * X_reduced.var(ddof=1) / trace
    unimodal = (phi >= 1.)
    return unimodal, p_value(phi, n_obs, dim), phi


def batch_folding_test_cpp(X):
    """
    Perform statically the folding test of unimodality (using c++ library)

    Parameters
    ----------

    X: numpy.ndarray
        d by n matrix (n observations in dimension d)
    """
    try:
        n, p = X.shape
    except:
        X = X.reshape(1, len(X))
        n, p = X.shape

    if n > p:  # if lines are observations, we transpose it
        X = X.T
        dim = p
        n_obs = n
    else:
        dim = n
        n_obs = p

    unimodal = c_bool(True)
    p_val = c_double(0.)
    phi = c_double(0.)
    time = c_long(0)
    LIBFOLDING.batch_folding_test(X, dim, n_obs, byref(unimodal),
                                  byref(p_val), byref(phi), byref(time))
    return unimodal.value, p_val.value, phi.value, time.value


#N = 20000
#X = np.random.normal(0, 1, N)
#print(batch_folding_test_cpp(X))
#print(batch_folding_test(X))

#sf = StreamFolding(N)
#for x in X.reshape(-1, 1):
#    sf.update(x)

#print(sf.folding_test())

#N = 200000
#X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], N).T
#X = np.random.normal(0, 1, (2, N))
#print(batch_folding_test(X))
#print(batch_folding_test_cpp(X))
