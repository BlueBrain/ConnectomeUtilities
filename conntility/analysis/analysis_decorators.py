import pandas
import numpy

from ..circuit_models.neuron_groups import group_with_config, filter_with_config
from ..circuit_models.neuron_groups.grouping_config import filter_config_to_dict

from ..io.logging import get_logger

LOG = get_logger("DECORATORS")

def __submatrix_presyn__(matrix):
    return lambda x: matrix[x.values]

def __submatrix_postsyn__(matrix):
    return lambda x: matrix[:, x.values]

def __submatrix_population__(matrix):
    return lambda x: matrix[numpy.ix_(x.values, x.values)]


def grouped_presyn_by_grouping_config(grp_cfg):
    """
    Perform an analysis separately on submatrices corresponding to presynaptic groups of neurons.
    That is: for a given population A it is executed on M[A, :]
    """
    return _grouped_by_grouping_config(grp_cfg, __submatrix_presyn__)


def grouped_postsyn_by_grouping_config(grp_cfg):
    """
    Perform an analysis separately on submatrices corresponding to postsynaptic groups of neurons.
    That is: for a given population A it is executed on M[:, A]
    """
    
    return _grouped_by_grouping_config(grp_cfg, __submatrix_postsyn__)


def grouped_subpop_by_grouping_config(grp_cfg):
    """
    Perform an analysis separately on submatrices corresponding to subpopulations of neurons.
    That is: for a given population A it is executed on M[numpy.ix_(A, A)]
    """
    return _grouped_by_grouping_config(grp_cfg, __submatrix_population__)


grouped_by_grouping_config = grouped_subpop_by_grouping_config


def _grouped_by_grouping_config(grp_cfg, submatrix_func):
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            grouped = group_with_config(nrn_df, grp_cfg)
            submatrices = grouped["__index__"].groupby(grouped.index.names).apply(submatrix_func(matrix))
            idxx = grouped.index.to_frame().drop_duplicates()
            if isinstance(submatrices.index, pandas.MultiIndex):
                colnames = idxx.columns
                idxx = list(map(tuple, idxx.values))
                out_index = pandas.MultiIndex.from_tuples(idxx, names=colnames)
            else:
                idxx = idxx.values[:, 0]
                out_index = pandas.Index(idxx, name=grouped.index.name)

            ret = [analysis_function(submatrices[ix], grouped.loc[ix], *args, **kwargs) for ix in idxx]
            if numpy.all([isinstance(_res, pandas.Series) for _res in ret]):
                ret = pandas.concat(ret, axis=0, keys=out_index, names=out_index.names, copy=False)
            else:
                ret = pandas.Series([analysis_function(submatrices[ix], grouped.loc[ix], *args, **kwargs)
                                    for ix in idxx], index=out_index)
            return ret
        return out_function
    return decorator


def __index_from_filter_configs(lst_fltr_cfg):
    if numpy.all(["name" in fltr_cfg for fltr_cfg in lst_fltr_cfg]):
        return pandas.DataFrame.from_records([
            (fltr_cfg["name"], ) for fltr_cfg in lst_fltr_cfg
        ], columns=["group name"])

    midx = [filter_config_to_dict(fltr_cfg) for fltr_cfg in lst_fltr_cfg]
    if numpy.any([numpy.sum([len(_v) for _v in _idx.values()]) > 50 for _idx in midx]):
        LOG.warn("""Automatically generated index for grouping excessively long.
        Think about explicitly specifying name="some_name" for filtered groups.""")

    return pandas.DataFrame.from_records(midx).astype(str)


def grouped_presyn_by_filtering_config(lst_fltr_cfg, *args):
    if len(args) > 0:  # In case someone mishandles how the arguments are given.
        lst_fltr_cfg = [lst_fltr_cfg] + args
    return _grouped_by_filtering_config(lst_fltr_cfg, __submatrix_presyn__)


def grouped_postsyn_by_filtering_config(lst_fltr_cfg, *args):
    if len(args) > 0:  # In case someone mishandles how the arguments are given.
        lst_fltr_cfg = [lst_fltr_cfg] + args
    return _grouped_by_filtering_config(lst_fltr_cfg, __submatrix_postsyn__)


def grouped_population_by_filtering_config(lst_fltr_cfg, *args):
    if len(args) > 0:  # In case someone mishandles how the arguments are given.
        lst_fltr_cfg = [lst_fltr_cfg] + args
    return _grouped_by_filtering_config(lst_fltr_cfg, __submatrix_population__)


grouped_by_filtering_config = grouped_population_by_filtering_config


def _grouped_by_filtering_config(lst_fltr_cfg, matrix_func):
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            midx = __index_from_filter_configs(lst_fltr_cfg)
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            groups = [filter_with_config(nrn_df, fltr_cfg) for fltr_cfg in lst_fltr_cfg]

            matrix_lo = matrix_func(matrix)
            ret = [analysis_function(
                matrix_lo(grp["__index__"]),
                grp, *args, **kwargs
                ) for grp in groups]
            if numpy.all([isinstance(_res, pandas.Series) for _res in ret]):
                ret = pandas.concat(ret, axis=0, copy=False,
                keys=map(tuple, midx.values), names=midx.columns.values.tolist())
            else:
                ret = pandas.Series(ret, index=pandas.MultiIndex.from_frame(midx))
            return ret
        return out_function
    return decorator

def control_by_randomization(randomization, n_randomizations=10, **rand_kwargs):
    if hasattr(randomization, "apply"):
        func = randomization.apply
        rand_name = randomization.name
    else:
        func = randomization
        rand_name = "randomized"
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            base_val = analysis_function(matrix, nrn_df, *args, **kwargs)
            cmp_vals = [
                analysis_function(
                    func(matrix, nrn_df, **rand_kwargs),
                    nrn_df, *args, **kwargs
                ) for _ in range(n_randomizations)
            ]
            if isinstance(base_val, pandas.Series):
                cmp_vals = pandas.concat(cmp_vals, axis=1).mean(axis=1)
                return pandas.concat(
                    [base_val, cmp_vals], axis=0, copy=False,
                    keys=["data", rand_name], names=["Control"]
                )
            cmp_vals = numpy.nanmean(cmp_vals)
            return pandas.Series([base_val, cmp_vals], index=["data", rand_name])
        return out_function
    return decorator


def control_by_random_sample(con_mat_obj, control_property, n_randomizations=10, sample_func=None, **rand_kwargs):
    from .. import ConnectivityMatrix
    assert isinstance(con_mat_obj, ConnectivityMatrix), "This decorator must be used through ConnectivityMatrix.analyze!"
    ctrl_index = con_mat_obj.index(control_property)
    if sample_func is None:
        func = ctrl_index.random
    else:
        func = getattr(ctrl_index, sample_func)
    rand_name = "sampled_by_" + control_property
    
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            base_val = analysis_function(matrix, nrn_df, *args, **kwargs)
            cmp_vals = [
                analysis_function(
                    func(nrn_df.index.values, **rand_kwargs).matrix.tocsc(),
                    nrn_df, *args, **kwargs
                ) for _ in range(n_randomizations)
            ]
            if isinstance(base_val, pandas.Series):
                cmp_vals = pandas.concat(cmp_vals, axis=1).mean(axis=1)
                return pandas.concat(
                    [base_val, cmp_vals], axis=0, copy=False,
                    keys=["data", rand_name], names=["Control"]
                )
            cmp_vals = numpy.nanmean(cmp_vals)
            return pandas.Series([base_val, cmp_vals], index=["data", rand_name])
        return out_function
    return decorator


def for_bidirectional_connectivity():
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            matrix = matrix.astype(bool)
            bd_matrix = (matrix.astype(int) + matrix.transpose()) == 2
            return analysis_function(bd_matrix, nrn_df, *args, **kwargs)


def for_undirected_connectivity():
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            matrix = matrix.astype(bool)
            ud_matrix = (matrix + matrix.transpose())
            return analysis_function(ud_matrix, nrn_df, *args, **kwargs)
