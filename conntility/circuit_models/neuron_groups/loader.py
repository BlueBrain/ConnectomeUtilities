# Functionality for loading information about neurons in a Circuit
# Wrapper and additional functionality for Circuit.cells.get
import numpy
import pandas
import os

from .extra_properties import add_extra_properties
from .defaults import VIRTUAL_FIBERS_FN, GID, FIBER_GID, PROJECTION


def load_neurons(circ, properties, base_target=None, **kwargs):
    """
    Loads information about the neurons in a Circuit. Provides access to all properties
    that Circuit.cells.get loads, plus flat mapped locations with the following names:
    flat_x: Flat mapped x coordinate (pixel resolution)
    flat_y: Flat mapped y coordinate (pixel resolution)
    ss_flat_x: Supersampled flat mapped x coordinate (approximately in um)
    ss_flat_y: Supersampled flat mapped y coordinate (approximately in um)
    depth: Approximate cortical depth of a neuron in um

    Input:
    circ (bluepy.Circuit): The Circuit to load neurons from
    properties (list-like): List with names of properties to load. Can contain anything
    that Circuit.cells.get loads, plus any of the additional properties listed above.
    base_target (str, optional): The name of the neuron target to load. If not provided
    loads all neurons.
    """
    props_to_load = numpy.intersect1d(properties, list(circ.cells.available_properties))
    extra_props = numpy.setdiff1d(properties, list(circ.cells.available_properties))
    neurons = circ.cells.get(group=base_target, properties=props_to_load)
    neurons[GID] = neurons.index

    neurons = add_extra_properties(neurons, circ, extra_props, **kwargs)

    neurons.index = pandas.RangeIndex(len(neurons))
    return neurons


def load_projection_locations(circ, properties, projection_name, **kwargs):
    """
    Loads anatomical information about projection fibers innervating a Circuit.
    Provides access to the following properties:
    x, y, z: Coordinates of the fiber origin in atlas coordinates (um)
    u, v, w: Direction the fiber is assumed to grow and innervate along.
    apron: Whether it is an "apron" fiber
    Plus flat mapped locations with the following names:
    flat_x: Flat mapped x coordinate (pixel resolution)
    flat_y: Flat mapped y coordinate (pixel resolution)
    ss_flat_x: Supersampled flat mapped x coordinate (approximately in um)
    ss_flat_y: Supersampled flat mapped y coordinate (approximately in um)
    depth: Approximate cortical depth of a neuron in um

    Input:
    circ (bluepy.Circuit): The Circuit to load fibers from
    properties (list-like): List with names of properties to load. Can contain anything
    listed above.
    projection_name (str): The name of the projection to load. Must exist in the
    CircuitConfig file.
    """
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
    """
    Loads anatomical information about projection fibers innervating a Circuit.
    Provides access to the following properties:
    x, y, z: Coordinates of the fiber origin in atlas coordinates (um)
    u, v, w: Direction the fiber is assumed to grow and innervate along.
    apron: Whether it is an "apron" fiber
    Plus flat mapped locations with the following names:
    flat_x: Flat mapped x coordinate (pixel resolution)
    flat_y: Flat mapped y coordinate (pixel resolution)
    ss_flat_x: Supersampled flat mapped x coordinate (approximately in um)
    ss_flat_y: Supersampled flat mapped y coordinate (approximately in um)
    depth: Approximate cortical depth of a neuron in um

    Will load ALL projections listed in the CircuitConfig and return a DataFrame
    indexed by the name of the projections.

    Input:
    circ (bluepy.Circuit): The Circuit to load fibers from
    properties (list-like): List with names of properties to load. Can contain anything
    listed above.
    """
    proj_names = list(circ.config.get('projections', {}).keys())
    projs = pandas.concat([
        load_projection_locations(circ, properties, proj, **kwargs)
        for proj in proj_names
    ], keys=proj_names, names=[PROJECTION], axis=0)
    return projs.set_index(projs.index.droplevel(-1))
