# SPDX-License-Identifier: Apache-2.0
from . import library, analysis_decorators
from .analysis import SingleMethodAnalysisFromSource as Analysis
from .analysis import get_analyses


def neighborhood_indices(M, pre=True, post=True):
    import numpy, pandas
    assert M.shape[0] == M.shape[1]
    M = M.tocoo()
    base_df = pandas.DataFrame({"row": M.row, "col": M.col})
    
    idxx = pandas.Index(numpy.arange(M.shape[0]), name="center")
    nb_df = pandas.DataFrame({"neighbors": [[]] * M.shape[0]}, index=idxx)["neighbors"]
    if post:
        new_df = base_df.rename(columns={"row": "center", "col": "neighbors"}).groupby("center")["neighbors"].apply(list)
        nb_df = nb_df.combine(new_df, numpy.union1d, fill_value=[])
    if pre:
        new_df = base_df.rename(columns={"col": "center", "row": "neighbors"}).groupby("center")["neighbors"].apply(list)
        nb_df = nb_df.combine(new_df, numpy.union1d, fill_value=[])
    return nb_df.apply(lambda _x: _x.astype(int))
