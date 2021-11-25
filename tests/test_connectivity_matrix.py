import os
import pandas
import numpy

import bluepy

from scipy import sparse
from conntility import connectivity as test_module

CIRC_FN = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC_WM"
CIRC = bluepy.Circuit(CIRC_FN)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

numpy.random.seed(123)
m_dense = numpy.random.rand(10, 10) < 0.1
m_sparse = sparse.coo_matrix(m_dense)

props = pandas.DataFrame({
    "foo": ["bar"] * 5 + ["meh"] * 5,
    "depth": numpy.linspace(-200, 500, 10)
}, index=numpy.random.permutation(10))


def test_from_bluepy():
    load_cfg = {
        "loading": {
            "base_target": "Layer4", 
            "properties": ["x", "y", "z", "etype", "layer"]
            },
            "filtering": {"column": "etype", "value": "bIR"}
            }
    M = test_module.ConnectivityMatrix.from_bluepy(CIRC, load_config=load_cfg)
    ue = numpy.unique(M.etype)
    ul = numpy.unique(M.layer)
    assert len(ue) == 1 and ue[0] == "bIR"
    assert len(ul) == 1 and ul[0] == 4

def test_base_connectivity_matrix():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)

    assert M.matrix.nnz == 6
    assert M.subpopulation([4, 5, 6, 7]).matrix.nnz == 2
    assert M.dense_matrix.sum() == 6
    assert "depth" in M.vertex_properties
    assert "meh" in M.foo


def test_connectivity_matrix_node_indexing():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)
    assert len(M.index("depth").lt(300)) == 7
    assert len(M.index("depth").le(300)) == 7
    assert len(M.index("depth").gt(300)) == 3
    assert len(M.index("depth").ge(300)) == 3

    assert len(M.index("foo").eq("meh")) == 5
    assert len(M.index("foo").eq("meh")) == 5
    assert M.index("foo").isin(["bar", "not"]).matrix.nnz == 2

    numpy.random.seed()
    seed = numpy.random.randint(0, 100000)
    sel_gids = numpy.random.choice(M.gids[M.depth < M.depth[-1]], 4, replace=False)
    numpy.random.seed(seed)
    gids = M.index("depth").random_numerical_gids(sel_gids, n_bins=2)
    assert M.gids[-1] not in gids
    assert len(gids) == len(sel_gids)
    numpy.random.seed(seed)
    subpop = M.index("depth").random_numerical(sel_gids, n_bins=2)
    assert numpy.all(gids == subpop.gids)


def test_connectivity_matrix_edge_indexing():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)
    assert M.filter().lt(2).matrix.nnz == M.matrix.nnz
    assert M.filter().lt(1).matrix.nnz == 0

    numpy.random.seed(123)
    M.add_edge_property("for_testing", numpy.random.rand(len(M._edges)))
    assert M.default("for_testing").filter().lt(0.5).matrix.nnz == 3
