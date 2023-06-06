# SPDX-License-Identifier: Apache-2.0
# Functionality for defining groups of neurons according to their properties
import numpy
import pandas

from .tessellate import TriTille
from .defaults import GID

def group_by_properties(df_in, lst_props, prefix="idx-", replace=True):
    """
    Defines a group of neurons by their values of a list of specified properties.
    That is, each unique combination of values defines a separate group, so don't
    use this for float-valued properties.
    Groups are defined by the index of a DataFrame holding their properties.

    Input:
    df_in (pandas.DataFrame): Holds information on neurons or innervating fibers.
    One row per neuron / fiber, one column per property.
    lst_props (list-like): List of columns to use for grouping.
    prefix (str): A prefix to preprepend to the names of the used columns/properties
    for their use in the index. This is to avoid having columns in the frame and index
    with the same name, which can break .groupby statements.
    replace (bool, default=True): Whether to replace the existing index and just add
    additional columns to it

    Return:
    A copy of df_in with updated MultiIndex. The index defines the groups of neurons, 
    i.e. neurons / fibers with the same index are assumed to be part of the same group.
    If replace=True, a new index is created; otherwise columns are added to any existing
    index.
    """
    vals = df_in[lst_props]
    vals.columns = [prefix + col for col in vals.columns]
    if not replace:
        vals = pandas.concat([df_in.index.to_frame(), vals], axis=1)

    df_in = df_in.set_index(pandas.MultiIndex.from_frame(vals)).sort_index()
    return df_in


def group_by_binned_properties(df_in, lst_props, bins, prefix="binned-", replace=True):
    """
    Defines a group of neurons by their binned values of a list of specified properties.
    That is, values are binned and then each unique combination bin-values defines a
    separate group. Values must be binnable with numpy.digitize!
    Groups are defined by the index of a DataFrame holding their properties.

    Input:
    df_in (pandas.DataFrame): Holds information on neurons or innervating fibers.
    One row per neuron / fiber, one column per property.
    lst_props (list-like): List of columns to use for grouping.
    bins: Defines how to bin the values. Can be an integer, defining the number
    of bins to use, or a list, defining the bin borders directly. Can also be a dict
    index by the name of the column that it applies to (if you want to use different
    bins for different columns).
    prefix (str): A prefix to preprepend to the names of the used columns/properties
    for their use in the index. This is to avoid having columns in the frame and index
    with the same name, which can break .groupby statements.
    replace (bool, default=True): Whether to replace the existing index and just add
    additional columns to it

    Return:
    A copy of df_in with updated MultiIndex. The index defines the groups of neurons, 
    i.e. neurons / fibers with the same index are assumed to be part of the same group.
    If replace=True, a new index is created; otherwise columns are added to any existing
    index.
    """
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
        binned_vals = pandas.concat([df_in.index.to_frame(), binned_vals], axis=1, copy=False)

    df_in = df_in.set_index(pandas.MultiIndex.from_frame(binned_vals)).sort_index()
    return df_in


def group_by_grid(df_in, lst_props, radius, shape="hexagonally", prefix="grid-", replace=True):
    """
    Defines a group of neurons by splitting their locations in a coordinate system
    into a regular grid.
    That is, the coordinate values are split into a grid and then each cell of the grid
    defines a separate group.
    Groups are defined by the index of a DataFrame holding their properties.

    Input:
    df_in (pandas.DataFrame): Holds information on neurons or innervating fibers.
    One row per neuron / fiber, one column per property.
    lst_props (list-like): List of columns holding the coordinates to use for grouping.
    Must have length 2, i.e. this only works for two-dimensional grids!
    radius (float): Defines the resolution of the grid. In the same units as the 
    coordinates given.
    shape (str, default: hexagonally): Defines the shape of the grid. Must one one of:
       - rhombically
       - triangularly
       - hexagonally
    prefix (str): A prefix to preprepend to the names of the used columns/properties
    for their use in the index. This is to avoid having columns in the frame and index
    with the same name, which can break .groupby statements.
    replace (bool, default=True): Whether to replace the existing index and just add
    additional columns to it

    Return:
    A copy of df_in with updated MultiIndex. The index defines the groups of neurons, 
    i.e. neurons / fibers with the same index are assumed to be part of the same group.
    If replace=True, a new index is created; otherwise columns are added to any existing
    index.
    """
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
    """
    "Flips" the representation of a neuron group. I.e. instead of the index specifying the group,
    it will be indexed by one of the data columns (such as "gid") and the values will specify
    the group.
    Input:
    df_in: A pandas.DataFrame of neuron of fiber properties with groups defined by a MultiIndex.
    lst_values (list-like, optional): List of columns of the index of df_in to use as the new
    values of the output. If not provided, then all columns will be used.
    index (str, default: "gid"): Name of the column to use as the new index.
    contract_values (bool, default=False): If False, then the output will be a DataFrame with the
    columns specified by lst_values. If True, then it will be contracted to a Series by casting
    the columns to str and then concatenating them.
    categorical (bool, default=False): Only used in conjunction with contract_values=True. If True,
    then the values of the output will be a pandas.Categorical, which may be faster for some purposes.
    """
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
    """
    Counts the amount of overlap in two neuron grouping schemes. For example: scheme A classifies according
    to mtype, scheme two according to etype. Then this yields the amount of neurons with a given mtype AND etype.
    That would be equivalent to grouping neurons according to [mtype, etype] and then doing .value_counts on the
    index, so I guess this is not that useful.
    """
    def prepare_overlap(df_pre):
        def execute_overlap(df_post):
            return len(numpy.intersect1d(df_pre[in_column], df_post[in_column]))

        return execute_overlap

    res = df_one.groupby(df_one.index.names).apply(
        lambda df_pre:
        df_two.groupby(df_two.index.names).apply(prepare_overlap(df_pre))
    )
    return res
