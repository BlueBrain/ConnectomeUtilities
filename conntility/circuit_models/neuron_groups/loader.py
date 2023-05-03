# Functionality for loading information about neurons in a Circuit
# Wrapper and additional functionality for Circuit.cells.get
import numpy
import pandas
import os

from .extra_properties import add_extra_properties
from .defaults import VIRTUAL_FIBERS_FN, GID, FIBER_GID, PROJECTION, DEFAULT_EDGES


def load_neurons(circ, properties, base_target=None, node_population=None, **kwargs):
    """
    Loads information about the neurons in a Circuit. Provides access to all properties
    that Circuit.cells.get loads, plus flat mapped locations with the following names:
    flat_x: Flat mapped x coordinate (pixel resolution)
    flat_y: Flat mapped y coordinate (pixel resolution)
    ss_flat_x: Supersampled flat mapped x coordinate (approximately in um)
    ss_flat_y: Supersampled flat mapped y coordinate (approximately in um)
    depth: Approximate cortical depth of a neuron in um

    Input:
    circ (bluepysnap.Circuit): The Circuit to load neurons from
    properties (list-like): List with names of properties to load. Can contain anything
    that Circuit.cells.get loads, plus any of the additional properties listed above.
    base_target (str, optional): The name of the neuron target to load. If not provided
    loads all neurons.
    node_population (str, optional): The name of the node population to load from. Currently,
    in multi-population circuits, it is likely required to specify the population.
    """
    props_to_load = numpy.intersect1d(properties, list(circ.nodes.property_names))
    extra_props = numpy.setdiff1d(properties, list(circ.nodes.property_names))
    node = circ.nodes
    if node_population is not None:
        neurons = node[node_population].get(group=base_target, properties=props_to_load)
        neurons.index.name = GID  # Until my pull request is merged.
    else:
        try:
            neurons = node.get(group=base_target, properties=props_to_load)
        except Exception as err:
            if str(err).startswith("Same property"):
                raise ValueError("In multi-population circuits, must use node_population= kwarg")
            raise
    neurons = neurons.reset_index()

    neurons = add_extra_properties(neurons, circ, extra_props, **kwargs)
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
    circ (bluepysnap.Circuit): The Circuit to load fibers from
    properties (list-like): List with names of properties to load. Can contain anything
    listed above.
    projection_name (str): The name of the projection to load. Must exist in the
    Sonata config / edges file.
    """
    from .sonata_extensions import projection_fiber_info

    vfib_file = projection_fiber_info(circ, projection_name)
    if vfib_file is None: raise RuntimeError("Cannot find virtual fiber info for the selected projection!")
    if not os.path.isfile(vfib_file):
        raise RuntimeError("Virtual fiber info expected at {0} but not present!".format(vfib_file))

    vfib = pandas.read_csv(vfib_file)

    primary_props = vfib.columns.values
    props_to_load = [FIBER_GID] + properties
    extra_props = numpy.setdiff1d(properties, primary_props)
    vfib = add_extra_properties(vfib, circ, extra_props, **kwargs)[props_to_load]
    # vfib[PROJECTION] = projection_name

    return vfib


def load_all_projection_locations(circ, properties, proj_names=None, include_extention=True, **kwargs):
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

    The list of projections to load can be provided, otherwise ALL projections listed 
    in the CircuitConfig will be considered. A DataFrame indexed by the name of the 
    projections is returned.

    Input:
    circ (bluepysnap.Circuit): The Circuit to load fibers from
    properties (list-like): List with names of properties to load. Can contain anything
    listed above.
    """
    if proj_names is None:
        proj_names = [_x for _x in circ.edges.keys() if _x != DEFAULT_EDGES]
        if include_extention:
            from .sonata_extensions import projection_list
            proj_names = proj_names + projection_list(circ)
    projs = pandas.concat([
        load_projection_locations(circ, properties, proj, **kwargs)
        for proj in proj_names
    ], keys=proj_names, names=[PROJECTION], axis=0)
    return projs.set_index(projs.index.droplevel(-1))
