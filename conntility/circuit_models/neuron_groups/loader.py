import numpy
import pandas
import os

from .extra_properties import add_extra_properties
from .defaults import VIRTUAL_FIBERS_FN, GID, FIBER_GID, PROJECTION


def load_neurons(circ, properties, base_target=None, **kwargs):
    props_to_load = numpy.intersect1d(properties, list(circ.cells.available_properties))
    extra_props = numpy.setdiff1d(properties, list(circ.cells.available_properties))
    neurons = circ.cells.get(group=base_target, properties=props_to_load)
    neurons[GID] = neurons.index

    neurons = add_extra_properties(neurons, circ, extra_props, **kwargs)

    neurons.index = pandas.RangeIndex(len(neurons))
    return neurons


def load_projection_locations(circ, properties, projection_name, **kwargs):
    """Reads .csv file saved in the same directory as the sonata file of the projection
    to get locations and directions of the virtual fibers"""
    projection_fn = circ.projection(projection_name).metadata["Path"]
    vfib_file = os.path.join(os.path.split(os.path.abspath(projection_fn))[0], VIRTUAL_FIBERS_FN)
    if not os.path.isfile(vfib_file):
        raise RuntimeError("Cannot find virtual fiber info for the selected projection!")
    vfib = pandas.read_csv(vfib_file)

    primary_props = vfib.columns.values
    props_to_load = [FIBER_GID] + properties
    extra_props = numpy.setdiff1d(properties, primary_props)
    vfib = add_extra_properties(vfib, circ, extra_props, **kwargs)[props_to_load]
    # vfib[PROJECTION] = projection_name

    return vfib


def load_all_projection_locations(circ, properties, **kwargs):
    proj_names = list(circ.config.get('projections', {}).keys())
    projs = pandas.concat([
        load_projection_locations(circ, properties, proj, **kwargs)
        for proj in proj_names
    ], keys=proj_names, names=[PROJECTION], axis=0)
    return projs.set_index(projs.index.droplevel(-1))
