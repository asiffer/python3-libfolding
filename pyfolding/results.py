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

from typing import NewType


class FTUResults:
    def __init__(self):
        self.n_obs = None
        self.dim = None
        self.folding_pivot = None
        self.cov = None
        self.folded_variance = None
        self.folding_ratio = None
        self.folding_statistics = None
        self.p_value = None
        self.time = None
        self.message = None

    def __str__(self):
        s = """
           #observations: {:d}
                     dim: {:d}
                    Φ(X): {:.9f}
                    φ(X): {:.9f}
         folded variance: {:.9f}
           folding pivot: {}
                 p-value: {:.9f}
                    time: {:.9f}
                 message: {}
        """.format(self.n_obs,
                   self.dim,
                   self.folding_statistics,
                   self.folding_ratio,
                   self.folded_variance,
                   self.folding_pivot,
                   self.p_value,
                   self.time,
                   self.message,
                   )
        return s


# FTU result type
Results = NewType('Results', FTUResults)
