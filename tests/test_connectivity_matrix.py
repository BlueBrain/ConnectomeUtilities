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


def test_subpopulation():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)
    idx = numpy.random.choice(len(M), int(0.8*len(M)), replace=False)
    idgids = M.gids[idx]
    assert (M.subpopulation(idgids).array == M.array[numpy.ix_(idx, idx)]).all()
    assert (M.subpopulation(idgids).array == M.subarray(idgids)).all()


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

    sel_gids = M.gids[[0, -1]]
    subpop = M.index("foo").random_categorical(sel_gids)
    assert len(subpop) == 2
    assert "bar" in subpop.foo and "meh" in subpop.foo


def test_connectivity_matrix_edge_indexing_and_adding():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)
    assert M.filter().lt(2).matrix.nnz == M.matrix.nnz
    assert M.filter().lt(1).matrix.nnz == 0

    numpy.random.seed(123)
    M.add_edge_property("for_testing", numpy.random.rand(len(M._edges)))
    Md = M.default("for_testing")
    assert len(Md.filter().lt(0.5)._edges) + len(Md.filter().ge(0.5)._edges) == len(M._edges)
    assert len(Md.filter().le(0.5)._edges) + len(Md.filter().gt(0.5)._edges) == len(M._edges)
    assert M.default("for_testing").filter().lt(0.5).matrix.nnz == 3

    int_property = numpy.random.randint(1, 6, size=M.array.shape)
    int_property[M.array == 0] = 0
    M.add_edge_property("category_testing", int_property)
    M.add_edge_property("category_testing2", sparse.coo_matrix(int_property))
    assert len(M.filter("category_testing").isin([1, 2, 3, 4, 5])._edges) == len(M._edges)
    fltrd_M = M.filter("category_testing").eq(1)
    assert (fltrd_M.edges["category_testing"] == 1).mean() == 1.0


def test_load_save():
    M = test_module.ConnectivityMatrix(m_sparse, vertex_properties=props)
    M.to_h5("test.h5", group_name="test_group", prefix="test_matrix")
    N = test_module.ConnectivityMatrix.from_h5("test.h5", group_name="test_group",
    prefix="test_matrix")
    assert (M.array == N.array).all()


def test_time_dependent_matrix():
    numpy.random.seed(123)
    row = numpy.random.randint(0, 100, 500)
    col = numpy.random.randint(0, 100, 500)
    rowcol = pandas.DataFrame({"row": row, "col": col}).drop_duplicates()
    L = len(rowcol)
    df = {
        "a": pandas.DataFrame(numpy.random.rand(L, 3), columns=[0.0, 10.0, 20.0]),
        "b": pandas.DataFrame(numpy.random.rand(L, 3), columns=[0.0, 10.0, 20.0])
    }

    M = test_module.TimeDependentMatrix(rowcol["row"].values, rowcol["col"].values,
                                        edge_properties=df)
    assert M.matrix.nnz == len(rowcol)
    assert M.edges.shape[1] == 2
    assert M._time == 0.0
    assert M.filter().lt(0.2).matrix.nnz == 82
    assert M.at_time(10.0).filter().lt(0.2).matrix.nnz == 98
    Mb = M.default("b")
    assert Mb._time == M._time
    assert Mb.at_time(20.0).filter().lt(0.2).matrix.nnz == 96

