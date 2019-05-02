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

from .libfolding import library, Matrix
from .results import FTUResults, Results

from math import log, sqrt, exp
from ctypes import c_double, c_int, c_bool, byref
from scipy.optimize import brenth, bisect, minimize
from numpy import float64
import numpy as np
import time


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
    >>> for x in U:
    ...     sf.update(x)

    >>> sf.mean()
    array([-0.04876098,  0.03932685])
    >>> sf.folding_pivot()
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
        self._sf_ptr = library.sf_new(depth)
        self._depth = depth
        self._dim = None

    def is_initialized(self):
        """
        Check if the object has been initialized (i.e. if the statistics has
        been computed at least once)
        """
        return bool(library.sf_is_initialized(self._sf_ptr))

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
            library.sf_update(self._sf_ptr, new_data, self._dim, True, False)
        elif obs_dim == self._dim:
            library.sf_update(self._sf_ptr, new_data, self._dim, True, False)
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
        library.sf_mean(self._sf_ptr, mean_data)
        return mean_data

    def s2star(self):
        """
        Compute the pivot s2*

        Returns
        -------
        folding_pivot: numpy.ndarray
            the folding pivot (shape (d,))
        """
        raise BaseException("'s2star' is deprecated, use 'folding_pivot' instead")

    def folding_pivot(self) -> Matrix:
        """
        Compute the pivot s2*

        Returns
        -------
        folding_pivot: numpy.ndarray
            the folding pivot (shape (d,))
        """
        folding_pivot = np.ones(self._dim)
        library.sf_s2star(self._sf_ptr, folding_pivot)
        return folding_pivot

    def cov(self) -> Matrix:
        """
        Compute the covariance of the stored data

        Returns
        -------
        cov_data: numpy.ndarray
            the covariance matrix (shape (d,d))
        """
        cov_data = np.ones((self._dim, self._dim))
        library.sf_cov(self._sf_ptr, cov_data)
        return cov_data

    def dump(self) -> Matrix:
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
        library.sf_dump(self._sf_ptr, data)
        return data

    def folding_test(self) -> Results:
        """
        Perform the folding test of unimodality within the current window

        Returns
        -------

        FTUResults
        """
        unimodal = c_bool(True)
        p_val = c_double(0.)
        phi = c_double(0.)

        t0 = time.time()
        library.sf_folding_test(self._sf_ptr, byref(unimodal), byref(p_val), byref(phi))
        t1 = time.time()

        # building the result
        ftu_results = FTUResults()
        ftu_results.folding_pivot = self.folding_pivot()
        ftu_results.cov = self.cov()
        ftu_results.folding_statistics = phi.value
        ftu_results.folding_ratio = ftu_results.folding_statistics / pow(1 + self._dim, 2)
        ftu_results.folded_variance = ftu_results.folding_ratio * np.trace(ftu_results.cov)
        ftu_results.p_value = p_val.value
        ftu_results.message = ""
        ftu_results.time = t1 - t0
        return ftu_results
