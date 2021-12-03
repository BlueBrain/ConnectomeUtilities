import pandas
import numpy

from ..circuit_models.neuron_groups import group_with_config, filter_with_config
from ..circuit_models.neuron_groups.grouping_config import filter_config_to_dict

from ..io.logging import get_logger

LOG = get_logger("DECORATORS")

def grouped_by_grouping_config(grp_cfg):
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            grouped = group_with_config(nrn_df, grp_cfg)
            submatrices = grouped["__index__"].groupby(grouped.index.names).apply(lambda x: matrix[numpy.ix_(x.values, x.values)])
            idxx = grouped.index.to_frame().drop_duplicates()
            if isinstance(submatrices.index, pandas.MultiIndex):
                idxx = list(map(tuple, idxx.values))
                out_index = pandas.MultiIndex.from_tuples(idxx)
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


def grouped_by_filtering_config(lst_fltr_cfg, *args):
    if len(args) > 0:  # In case someone mishandles how the arguments are given.
        lst_fltr_cfg = [lst_fltr_cfg] + args
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            midx = __index_from_filter_configs(lst_fltr_cfg)
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            groups = [filter_with_config(nrn_df, fltr_cfg) for fltr_cfg in lst_fltr_cfg]

            ret = [analysis_function(
                matrix[numpy.ix_(grp["__index__"].values, grp["__index__"].values)],
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
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            if hasattr(randomization, "apply"):
                func = randomization.apply
                rand_name = randomization.name
            else:
                func = randomization
                rand_name = "randomized"
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
