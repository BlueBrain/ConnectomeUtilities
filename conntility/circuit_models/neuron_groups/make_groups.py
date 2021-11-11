import numpy
import pandas
import os

from .extra_properties import add_extra_properties
from .tessellate import TriTille


VIRTUAL_FIBERS_FN = "virtual-fibers.csv"
VIRTUAL_FIBERS_MASK = "apron"
VIRTUAL_FIBERS_GIDS = "sgid"
VIRTUAL_FIBERS_XYZ = ["x", "y", "z"]
VIRTUAL_FIBERS_UVW = ["u", "v", "w"]


def load_neurons(circ, properties, base_target=None, **kwargs):
    props_to_load = numpy.intersect1d(properties, list(circ.cells.available_properties))
    extra_props = numpy.setdiff1d(properties, list(circ.cells.available_properties))
    neurons = circ.cells.get(group=base_target, properties=props_to_load)
    neurons["gid"] = neurons.index

    neurons = add_extra_properties(neurons, circ, extra_props, **kwargs)

    neurons.index = pandas.RangeIndex(len(neurons))
    return neurons


def load_projection_locations(circ, projection_name):
    """Reads .csv file saved in the same directory as the sonata file of the projection
    to get locations and directions of the virtual fibers"""
    projection_fn = circ.projection(projection_name).metadata["Path"]
    vfib_file = os.path.join(os.path.split(os.path.abspath(projection_fn))[0], VIRTUAL_FIBERS_FN)
    if not os.path.isfile(vfib_file):
        raise RuntimeError("Cannot find virtual fiber info for the selected projection!")
    vfib = pandas.read_csv(vfib_file)
    return vfib


def group_by_properties(df_in, lst_props, prefix="idx-", replace=True):
    vals = df_in[lst_props]
    vals.columns = [prefix + col for col in vals.columns]
    if not replace:
        vals = pandas.concat([df_in.index.to_frame(), vals])

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
        binned_vals = pandas.concat([df_in.index.to_frame(), binned_vals])

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

