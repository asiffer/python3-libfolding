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
from .pvalues import p_value, decision_bound

import numpy as np
from scipy.optimize import minimize
from ctypes import c_long, c_double, c_int, c_bool, byref
import time


def ftu(X: Matrix) -> Results:
    """
    Parameters
    ----------

    X: numpy.ndarray
        n by d matrix (n observations in dimension d)
    """
    ftu_results = FTUResults()
    t0 = time.time()

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ftu_results.n_obs, ftu_results.dim = X.shape
    ftu_results.cov = np.cov(X.T).reshape(ftu_results.dim,
                                          ftu_results.dim)  # cov(X)

    X_square_norm = (X * X).sum(axis=1)  # |X|²

    try:
        # cov(X,|X|²)
        cov_norm = np.cov(X.T,
                          X_square_norm)[:-1, -1].reshape(-1, 1)
        ftu_results.folding_pivot = 0.5 * np.linalg.solve(ftu_results.cov, cov_norm).flatten()
    except np.linalg.linalg.LinAlgError:
        # numerical optimization
        ftu_results.folding_pivot = minimize(lambda s: np.power(
            X.T - s, 2).sum(axis=1).var(),
            x0=X.mean(axis=1)).x.flatten()

    X_reduced = np.linalg.norm(X - ftu_results.folding_pivot, axis=1)  # |X-s*|
    ftu_results.folded_variance = X_reduced.var()
    ftu_results.folding_ratio = ftu_results.folded_variance / np.trace(ftu_results.cov)
    ftu_results.folding_statistics = pow(1. + ftu_results.dim, 2) * ftu_results.folding_ratio
    ftu_results.p_value = p_value(ftu_results.folding_statistics,
                                  ftu_results.n_obs,
                                  ftu_results.dim)
    ftu_results.time = time.time() - t0
    ftu_results.message = ''
    return ftu_results


def ftu_cpp(X: Matrix) -> Results:
    """
    Parameters
    ----------

    X: numpy.ndarray
        n by d matrix (n observations in dimension d)
    """
    ftu_results = FTUResults()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ftu_results.n_obs, ftu_results.dim = X.shape
    ftu_results.cov = np.cov(X.T).reshape(ftu_results.dim,
                                          ftu_results.dim)  # cov(X)
    X = np.ascontiguousarray(X)
    # Preparing C++ call
    unimodal = c_bool(True)
    p_val = c_double(0.)
    phi = c_double(0.)
    time = c_long(0)
    s = np.ones(ftu_results.dim)
    library.batch_folding_test(X,
                               ftu_results.dim,
                               ftu_results.n_obs,
                               s,
                               byref(unimodal),
                               byref(p_val),
                               byref(phi),
                               byref(time))
    ftu_results.folding_pivot = s
    ftu_results.folding_statistics = phi.value
    ftu_results.folding_ratio = ftu_results.folding_statistics / pow(1. + ftu_results.dim, 2)
    ftu_results.folded_variance = ftu_results.folding_ratio * np.trace(ftu_results.cov)
    ftu_results.p_value = p_val.value

    ftu_results.time = time.value / 1e6
    ftu_results.message = ''
    return ftu_results


def FTU(X: Matrix, routine: str = 'python') -> Results:
    if routine.lower() in ['python', 'py']:
        return ftu(X)
    elif routine.lower() in ['c++', 'cpp']:
        return ftu_cpp(X)
    else:
        raise ValueError("Unknown routine (try 'python' or 'cpp')")
