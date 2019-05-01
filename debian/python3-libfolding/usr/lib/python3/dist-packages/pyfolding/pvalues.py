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

from math import log, sqrt, exp
from scipy.optimize import brenth

# constants to compute FTU p_values
PVAL_A = 0.4785
PVAL_B = 0.1946
PVAL_C = 2.0287


def decision_bound(p_value: float, n: int, d: int) -> float:
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


def p_value(phi: float, n: int, d: int) -> float:
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
        def obj_fun(p): return (abs(phi - 1.) - decision_bound(1 - p, n, d))
        p_val = brenth(obj_fun, 0., 1.)
    except BaseException:
        p_val = exp(-abs(phi - 1.) * sqrt(n) / (PVAL_C + log(d)))
    return p_val
