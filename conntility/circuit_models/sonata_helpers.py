import numpy

from .neuron_groups.sonata_extensions import projection_list

LOCAL_CONNECTOME = "local"


def find_sonata_connectome(circ, connectome, return_sonata_file=True):
    """
    Returns the sonata connectome associated with a named projection; or the default "local" connectome.
    Input:
    circ (bluepysnap.Circuit)
    connectome (str): Name of the projection to look up. Use "local" for the default local (i.e. touch-based) connectome
    return_sonata_file (optional): If true, returns the path of the .h5 file. Else returns a bluepy.Connectome.
    """
    if connectome == LOCAL_CONNECTOME: 
        connectome = local_connectome_for(circ, nonvirtual_node_population(circ))
    if return_sonata_file:
        if connectome in circ.edges:
            return circ.edges[connectome].h5_filepath
        proj_list = projection_list(circ, return_filename_dict=True)
        if connectome in proj_list: 
            return proj_list[connectome]
    if connectome in circ.edges:
        return circ.edges[connectome]
    proj_list = projection_list(circ)
    if connectome in proj_list:
        raise RuntimeError("Old style connectome {0} can only be returned as filename reference".format(connectome))
    raise RuntimeError("Connectome {0} not found!".format(connectome))


def get_connectome_shape(circ, connectome):
    if connectome == LOCAL_CONNECTOME: 
        connectome = local_connectome_for(circ, nonvirtual_node_population(circ))
    if connectome in circ.edges:
        pop = circ.edges[connectome]
        return (pop.source.size, pop.target.size)
    return (circ.nodes.size, circ.nodes.size)


def local_connectomes_for(circ, node_pop):
    hits = [k for k, v in circ.edges.items()
            if v.source.name == node_pop
            and v.target.name == node_pop]
    return hits


def local_connectome_for(circ, node_pop):
    hits = local_connectomes_for(circ, node_pop)
    assert len(hits) > 0, "No local connectome for {0}".format(node_pop)
    if len(hits) > 1:
        print("Warning: More than one local connectome for {0}. Returning the first: {1}".format(node_pop,
                                                                                                 hits[0]))
    return hits[0]


def projection_connectomes_for(circ, node_pop):
    hits = [k for k, v in circ.edges.items()
            if v.source.name != node_pop
            and v.target.name == node_pop]
    return hits


def nonvirtual_node_populations(circ):
    hits = [k for k, v in circ.nodes.items()
            if v.config.get("type", "virtual") != "virtual"]
    return hits


def nonvirtual_node_population(circ):
    hits = nonvirtual_node_populations(circ)
    assert len(hits) > 0, "No non-virtual population found!"
    idx = numpy.argsort([circ.nodes[_x].size for _x in hits])[-1]
    if len(hits) > 1:
        print("Warning: More than one non-virtual nodes population. Returning the largest: {0}".format(hits[idx]))
    return hits[idx]
