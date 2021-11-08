import numpy
import pandas

from .extra_properties import add_extra_properties


def load_neurons(circ, properties, base_target=None, **kwargs):
    props_to_load = numpy.intersect1d(properties, list(circ.cells.available_properties))
    extra_props = numpy.setdiff1d(properties, list(circ.cells.available_properties))
    neurons = circ.cells.get(group=base_target, properties=props_to_load)
    neurons["gid"] = neurons.index

    neurons = add_extra_properties(neurons, circ, extra_props, **kwargs)

    neurons.index = pandas.RangeIndex(len(neurons))
    return neurons


def group_by_properties(df_in, lst_props, inplace=True):
    idxx = pandas.MultiIndex.from_frame(df_in[lst_props])
    idxx.names = ["idx-" + x for x in idxx.names]
    if not inplace:
        df_in = df_in.copy()
    df_in.index = idxx
    df_in = df_in.sort_index()
    return df_in


def group_by_binned_properties(df_in, lst_props, bins, inplace=True):
    def make_bins(values):
        if isinstance(bins, dict):
            b = bins[values.name]
        else:
            b = bins
        if not hasattr(b, "__iter__"):
            b = numpy.linspace(numpy.min(values), numpy.max(values) + 1E-12, b)
        return b
    binned_vals = df_in[lst_props].apply(lambda x: numpy.digitize(x, bins=make_bins(x)), axis=0)
    binned_vals.columns = ["binned-" + x for x in binned_vals.columns]
    if not inplace:
        df_in = df_in.copy()
    df_in.index = pandas.MultiIndex.from_frame(binned_vals)
    df_in = df_in.sort_index()
    return df_in


def group_by_square_grid(df_in, resolution=None, neurons_per_group=None):
    raise NotImplementedError()


def group_by_hex_grid(df_in, resolution=None, neurons_per_group=None):
    raise NotImplementedError()


def flip(df_in, lst_values=None, index="gid", contract_values=False, categorical=False):
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

