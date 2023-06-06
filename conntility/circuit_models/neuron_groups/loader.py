# SPDX-License-Identifier: Apache-2.0
# Functionality for loading information about neurons in a Circuit
# Wrapper and additional functionality for Circuit.cells.get
import numpy
import pandas
import os

from .extra_properties import add_extra_properties
from .defaults import GID, FIBER_GID, PROJECTION, DEFAULT_EDGES
from ..sonata_helpers import source_connectomes_for, target_connectomes_for


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


def load_source_locations(circ, properties, projection_name, 
                          base_target=None, **kwargs):
    """
    Loads anatomical information about source nodes of a connectome.
    Properties that can be loaded: See load_neurons.

    Args:
        circ (bluepysnap.Circuit): The Circuit to load fibers from
        properties (list-like): List with names of properties to load. See load_neurons.
        projection_name (str): The name of the connectome to load the source population
        of. Must be in circ.edges.
        base_target (optional): Name of a node_set. If provided, loads the intersection
        of the source population with the node set.
    """
    if projection_name not in circ.edges:
        raise ValueError("No connectome {0} found!".format(projection_name))
    nodepop = circ.edges[projection_name].source.name
    return load_neurons(circ, properties, base_target=base_target,
                        node_population=nodepop, **kwargs)

def load_target_locations(circ, properties, projection_name, 
                          base_target=None, **kwargs):
    """
    Loads anatomical information about target nodes of a connectome.
    Properties that can be loaded: See load_neurons.

    Args:
        circ (bluepysnap.Circuit): The Circuit to load fibers from
        properties (list-like): List with names of properties to load. See load_neurons.
        projection_name (str): The name of the connectome to load the target population
        of. Must be in circ.edges.
        base_target (optional): Name of a node_set. If provided, loads the intersection
        of the target population with the node set.
    """
    if projection_name not in circ.edges:
        raise ValueError("No connectome {0} found!".format(projection_name))
    nodepop = circ.edges[projection_name].target.name
    return load_neurons(circ, properties, base_target=base_target,
                        node_population=nodepop, **kwargs)


def load_all_source_locations(circ, properties, node_population,
                              base_target=None, **kwargs):
    """
    Loads anatomical information about nodes innervating a node_population.
    Properties that can be loaded: See load_neurons.

    Args:
        circ (bluepysnap.Circuit): The Circuit to load fibers from
        properties (list-like): List with names of properties to load. See load_neurons.
        node_population (str): The name of a node population. All node populations innervating
        this population through any EdgePopulation will be loaded.
        base_target (optional): Name of a node_set. If provided, loads the intersection
        of the above with the node set.
    """
    connectomes = source_connectomes_for(circ, node_population)
    return pandas.concat([
        load_source_locations(circ, properties, conn, base_target=base_target, **kwargs)
        for conn in connectomes
    ], axis=0)


def load_all_target_locations(circ, properties, node_population,
                              base_target=None, **kwargs):
    """
    Loads anatomical information about nodes innervated by a node_population.
    Properties that can be loaded: See load_neurons.

    Args:
        circ (bluepysnap.Circuit): The Circuit to load fibers from
        properties (list-like): List with names of properties to load. See load_neurons.
        node_population (str): The name of a node population. All node populations innervated by
        this population through any EdgePopulation will be loaded.
        base_target (optional): Name of a node_set. If provided, loads the intersection
        of the above with the node set.
    """
    connectomes = target_connectomes_for(circ, node_population)
    return pandas.concat([
        load_source_locations(circ, properties, conn, base_target=base_target, **kwargs)
        for conn in connectomes
    ], axis=0)
