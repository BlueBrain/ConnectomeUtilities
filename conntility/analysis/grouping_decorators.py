import pandas
import numpy

from ..circuit_models.neuron_groups import group_with_config, filter_with_config
from ..circuit_models.neuron_groups.grouping_config import filter_config_to_dict

def grouped_by_grouping_config(grp_cfg):
    def decorator(analysis_function):
        def out_function(nrn_df, matrix, *args, **kwargs):
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            grouped = group_with_config(nrn_df, grp_cfg)
            submatrices = grouped["__index__"].groupby(grouped.index.names).apply(lambda x: matrix[numpy.ix_(x.values, x.values)])
            idxx = grouped.index.to_frame().drop_duplicates()
            idxx = list(map(tuple, idxx.values))
            ret = pandas.Series([analysis_function(submatrices[ix], grouped.loc[ix], *args, **kwargs)
                                 for ix in idxx], index=pandas.MultiIndex.from_tuples(idxx))
            return ret
        return out_function
    return decorator

def grouped_by_filtering_config(lst_fltr_cfg):
    def decorator(analysis_function):
        def out_function(nrn_df, matrix, *args, **kwargs):
            midx = pandas.DataFrame.from_dict([filter_config_to_dict(fltr_cfg)
            for fltr_cfg in lst_fltr_cfg])
            nrn_df = pandas.concat([nrn_df, pandas.Series(range(len(nrn_df)),
            index=nrn_df.index, name="__index__")], copy=False, axis=1)
            groups = [filter_with_config(nrn_df, fltr_cfg) for fltr_cfg in lst_fltr_cfg]
            ret = pandas.Series([analysis_function(
                matrix[numpy.ix_(grp["__index__"].values, grp["__index__"].values)],
                grp, *args, **kwargs
            ) for grp in groups], index=midx)
            return ret
        return out_function
    return decorator

