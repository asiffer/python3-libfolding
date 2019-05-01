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
from numpy import float64
import numpy as np
from typing import NewType

# Matrix type: numpy array
Matrix = NewType('Matrix', np.ndarray)


# C-types corresponding to matrix
ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=float64, ndim=1, flags='CONTIGUOUS')
ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=float64, ndim=2, flags='CONTIGUOUS')


# libfolding interface
try:
    library = CDLL('libfolding.so')
except OSError as error:
    print(error)
    print("libfolding is probably not installed. Visit https://asiffer.github.io/libfolding/")

library.sf_new.argtypes = [c_size_t]
library.sf_new.restype = c_void_p

library.sf_update.argtypes = [c_void_p, ND_POINTER_1, c_size_t, c_bool, c_bool]

library.sf_is_initialized.argtypes = [c_void_p]
library.sf_is_initialized.restypes = [c_bool]

library.sf_folding_test.argtypes = [c_void_p,
                                    POINTER(c_bool),
                                    POINTER(c_double),
                                    POINTER(c_double)]

library.sf_mean.argtypes = [c_void_p, ND_POINTER_1]

library.sf_cov.argtypes = [c_void_p, ND_POINTER_2]

library.sf_s2star.argtypes = [c_void_p, ND_POINTER_1]

library.sf_dump.argtypes = [c_void_p, ND_POINTER_2]

# library.batch_folding_test.argtypes = [ND_POINTER_2, c_size_t, c_size_t,
#                                        POINTER(c_bool), POINTER(c_double),
#                                        POINTER(c_double), POINTER(c_long)]

library.batch_folding_test.argtypes = [ND_POINTER_2, c_size_t, c_size_t,
                                       ND_POINTER_1,
                                       POINTER(c_bool), POINTER(c_double),
                                       POINTER(c_double), POINTER(c_long)]
