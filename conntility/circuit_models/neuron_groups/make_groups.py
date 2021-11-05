import numpy
import pandas


def load_neurons(circ, properties):
    props_to_load = numpy.intersect1d(properties, list(circ.cells.available_properties))
    extra_props = numpy.setdiff1d(properties, list(circ.cells.available_properties))
    neurons = circ.cells.get(properties=props_to_load)
    neurons["gid"] = neurons.index

    for prop in extra_props:
        if prop == "flat_coordinates":
            from ...flatmapping import supersampled_neuron_locations
            flat_locs = supersampled_neuron_locations(circ)
            neurons = pandas.concat([neurons, flat_locs], axis=1)
        else:
            raise ValueError()

    neurons.index = pandas.RangeIndex(len(neurons))
    return neurons


def group_by_properties(df_in, lst_props):
    idxx = pandas.MultiIndex.from_frame(df_in[lst_props])
    idxx.names = ["idx-" + x for x in idxx.names]
    df_in.index = idxx
    df_in = df_in.sort_index()
    return df_in


def group_by_square_grid(df_in, resolution=None, neurons_per_group=None):
    raise NotImplementedError()


def group_by_hex_grid(df_in, resolution=None, neurons_per_group=None):
    raise NotImplementedError()
