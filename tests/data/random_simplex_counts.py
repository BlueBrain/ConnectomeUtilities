# SPDX-License-Identifier: Apache-2.0
"""
Returns just random numbers. Just for testing purposes!
"""
import pandas
from numpy import random

def simplex_counts(M, nrn):
    return pandas.Series(
        [M.shape[0], M.nnz] + [random.randint(1, x) for x in [300, 200, 50]],
        index=pandas.Index(range(5), name="dim")
    )

def random_scalar(M, nrn):
    return random.randint(0, 123)
