import numpy
import pandas

from .tessellate import TriTille
from .defaults import GID

def group_by_properties(df_in, lst_props, prefix="idx-", replace=True):
    vals = df_in[lst_props]
    vals.columns = [prefix + col for col in vals.columns]
    if not replace:
        vals = pandas.concat([df_in.index.to_frame(), vals], axis=1)

    df_in = df_in.set_index(pandas.MultiIndex.from_frame(vals)).sort_index()
    return df_in


def group_by_binned_properties(df_in, lst_props, bins, prefix="binned-", replace=True):
    def make_bins(values):
        if isinstance(bins, dict):
            b = bins[values.name]
        else:
            b = bins
        if not hasattr(b, "__iter__"):
            b = numpy.arange(numpy.min(values), numpy.max(values) + 1E-12, b)
        return b
    binned_vals = df_in[lst_props].apply(lambda x: numpy.digitize(x, bins=make_bins(x)), axis=0)
    binned_vals.columns = [prefix + col for col in binned_vals.columns]

    if not replace:
        binned_vals = pandas.concat([df_in.index.to_frame(), binned_vals], axis=1)

    df_in = df_in.set_index(pandas.MultiIndex.from_frame(binned_vals)).sort_index()
    return df_in


def group_by_grid(df_in, lst_props, radius, shape="hexagonally", prefix="grid-", replace=True):
    assert len(lst_props) == 2, "Need one column as x-coordinate and one as y-coordinate"
    tritilling = TriTille(radius)

    flat_locs = df_in[lst_props].rename(dict([(a, b) for a, b in zip(lst_props, ["x", "y"])]), axis=1)

    hexmap = getattr(tritilling, "bin_" + shape)(flat_locs, use_columns_row_indexing=False)
    grid = tritilling.locate_grid(hexmap)

    annotation = tritilling.annotate(grid, using_column_row=True)
    annotated_grid = grid.assign(subtarget=annotation.loc[grid.index])
    per_neuron_annotation = annotated_grid.loc[pandas.MultiIndex.from_frame(hexmap)]

    hexmap.columns = [prefix + col for col in hexmap.columns]
    per_neuron_annotation.columns = [prefix + col for col in per_neuron_annotation.columns]
    if not replace:
        hexmap = hexmap.reset_index()
    per_neuron_annotation = per_neuron_annotation.set_index(pandas.MultiIndex.from_frame(hexmap))

    df_in = df_in.set_index(per_neuron_annotation.index)
    df_in = pandas.concat([df_in, per_neuron_annotation], axis=1)
    return df_in


def flip(df_in, lst_values=None, index=GID, contract_values=False, categorical=False):
    if isinstance(df_in, pandas.DataFrame):
        df_in = df_in[index]
    idx_frame = df_in.index.to_frame()
    if lst_values is None:
        lst_values = idx_frame.columns.values
    idx_frame.index = df_in.values
    idx_frame = idx_frame[lst_values]

    if contract_values:
        new_name = ";".join(idx_frame.columns)
        idx_series = idx_frame.apply(lambda row: "_".join(map(str, row)), axis=1)
        idx_series.name = new_name
        if categorical:
            idx_series = pandas.Series(pandas.Categorical(idx_series), index=idx_series.index, name=idx_series.name)
        return idx_series
    return idx_frame


def count_overlap(df_one, df_two, in_column=GID):
    def prepare_overlap(df_pre):
        def execute_overlap(df_post):
            return len(numpy.intersect1d(df_pre[in_column], df_post[in_column]))

        return execute_overlap

    res = df_one.groupby(df_one.index.names).apply(
        lambda df_pre:
        df_two.groupby(df_two.index.names).apply(prepare_overlap(df_pre))
    )
    return res
