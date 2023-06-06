# SPDX-License-Identifier: Apache-2.0
import pandas
import numpy


def input_spikes(sim):
    from .sonata_helpers import simulated_nodes, resolve_node_set

    circ = sim.circuit
    spks = [_spks for _spks in sim.config["inputs"].values() if _spks["input_type"] == "spikes"]

    out = []
    sim_target = simulated_nodes(sim)
    for spk in spks:
        stim_target = resolve_node_set(circ, spk["node_set"])
        source = resolve_node_set(circ, spk["source"])

        A = sim_target.value_counts()
        B = stim_target.value_counts()
        target = (A & B).reset_index().set_index("count").loc[True].reset_index(drop=True)

        s = pandas.read_csv(spk["spike_file"], sep="\s+").rename(columns={"/scatter": "node_id"})["node_id"]
        out.append((s, source, target))
    return out


def input_innervation_from_matrix(spikes, matrix, gids_pre, t_win=None):
    """
    How strongly each neuron is innervated by the simulation input spikes
    Input:
    spikes (pandas.Series): Input spikes as given by the input_spikes(sim) function
    matrix (scipy.sparse matrix): Connection matrix of the projection that the input
    spikes use.
    gids_pre (list): List of the gids associated with each row (presynaptic id) of 
    matrix.
    t_win (tuple or list of length 2): Time window of interest.

    Returns:
    innervation: A numpy.array of the number of spikes received by each postsynaptic element of
    matrix. That is: len(innervation) == matrix.shape[1]

    Note:
    Meant for input spikes, but you can also use this to count how strongly each neuron
    is innervated by the recurrent activity (use output spikes for "spikes" and the
    internal connection matrix for "matrix")
    """
    from scipy import sparse
    assert sparse.issparse(matrix), "Input connection matrix must be sparse!"

    if t_win is not None:
        if hasattr(t_win[0], "__iter__"):  # TODO: Take this out!
            return pandas.Series(
                [input_innervation_from_matrix(spikes, matrix, gids_pre, t_win=t)
                 for t in t_win], name="innervation",
                index=pandas.Index(["{0}-{1}".format(*t) for t in t_win], name="t_win")
            )
        spikes = spikes[(spikes.index < t_win[1]) & (spikes.index >= t_win[0])]
    spk_count = spikes.value_counts()
    count_vec = numpy.zeros((len(gids_pre), 1))
    nz_spikes = numpy.in1d(gids_pre, spk_count.index)
    count_vec[nz_spikes, 0] = spk_count[gids_pre[nz_spikes]]

    innervation = numpy.array(matrix.multiply(count_vec).sum(axis=0)).flatten()

    return innervation


def _input_innervation(circ, s, source, target, t_win=None):
    from .connection_matrix import circuit_node_set_matrix

    sM, src, tgt = circuit_node_set_matrix(circ, source, target)
    res = input_innervation_from_matrix(s, sM, src["node_ids"].values, t_win=t_win)
    if t_win is None:
        tgt["input_spike_count"] = res
    else:
        for k, v in res.items(): tgt[k] = v
    return tgt.set_index(["population", "node_ids"])


def input_innervation(sim, t_win=None, sum=True):
    """
    How strongly each neuron is innervated by the simulation input spikes
    Input:
    sim (bluepysnap.Simulation)
    t_win (optional, list of tuples): List of time windows of interest. If not
    provided, the entire simulation is considered as one single time bin.
    sim (optional, default=True): If True, sum up the contributions of different
    spike replay blocks. Else return them individually in a list.
    
    Returns:
    innervation: A pandas.DataFrame with input spike counts in columns.
    Indexed by: 
       "population": name of the node population being innervated
       "node_ids": ids of the innervated neurons.
    Column names are either ["input_spike_count"] or the names of the time windows,
    if specified.
    """
    spks = input_spikes(sim)
    if len(spks) == 0: return None

    res = _input_innervation(sim, *spks[0], t_win=t_win)
    if sum:
        for spk in spks[1:]:
            res = res.add(_input_innervation(sim, *spk, t_win=t_win), fill_value=0)
    else:
        res = [res]
        for spk in spks[1:]:
            res.append(_input_innervation(sim, *spk, t_win=t_win))
    return res
