# SPDX-License-Identifier: Apache-2.0
import os
import numpy
import pandas

from conntility.analysis import library as test_module

TST_SZ = 100

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_embedding_simple():
    x = numpy.linspace(0, 1, TST_SZ)
    X, Y = numpy.meshgrid(x, x)
    D = 1 / (numpy.abs(X - Y) + .1)
    a, b, c = test_module.embed_pathway(D, n_components=50)
    assert c['n_components_auto'] < 8  # Should be < 5. But playing it safe
    assert c['lambdas'][0] / c['lambdas'][1] > 2
