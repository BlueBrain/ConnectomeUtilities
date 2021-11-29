import pandas
import numpy

from ..circuit_models.neuron_groups import group_with_config, filter_with_config
from ..circuit_models.neuron_groups.grouping_config import filter_config_to_dict

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

def grouped_by_filtering_config(lst_fltr_cfg):
    def decorator(analysis_function):
        def out_function(matrix, nrn_df, *args, **kwargs):
            midx = pandas.DataFrame.from_dict([filter_config_to_dict(fltr_cfg)
            for fltr_cfg in lst_fltr_cfg])
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            groups = [filter_with_config(nrn_df, fltr_cfg) for fltr_cfg in lst_fltr_cfg]

            ret = [analysis_function(
                matrix[numpy.ix_(grp["__index__"].values, grp["__index__"].values)],
                grp, *args, **kwargs
                ) for grp in groups]
            if numpy.all([isinstance(_res, pandas.Series) for _res in ret]):
                ret = pandas.concat(ret, axis=0, copy=False, keys=midx, names=midx.names)
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
