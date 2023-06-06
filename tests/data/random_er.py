# SPDX-License-Identifier: Apache-2.0
import numpy

from scipy import sparse


def random_er(mat, nrn):
    mat = mat.tocoo()
    mat.row = numpy.random.permutation(mat.row)
    mat.col = numpy.random.permutation(mat.col)
    return mat.tocsc()
