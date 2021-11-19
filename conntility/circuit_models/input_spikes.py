import pandas
import numpy
import os


def input_spikes(sim):
    # TODO: This fails if there's no input spikes
    def read_csv(path):
        data = pandas.read_csv(path, sep="\t")["/scatter"]
        data.name = "gid"
        data.index.name = "t"
        return data

    sim_path = sim.config._path
    sim_root = os.path.split(sim_path)[0]

    stim = [stim for stim in sim.config.typed_sections("Stimulus") if stim["Pattern"] == "SynapseReplay"]
    if len(stim) == 0:
        return pandas.Series([], index=pandas.Float64Index([], name="t"), name="gid", dtype=float)

    stim = [_stim["SpikeFile"] for _stim in stim]
    stim = [_stim if os.path.isabs(_stim) else os.path.join(sim_root, _stim) for _stim in stim]
    spks = pandas.concat([read_csv(_stim) for _stim in stim], axis=0)
    return spks


def input_innervation_from_matrix(spikes, matrix, gids_pre, t_win=None):
    from scipy import sparse
    assert isinstance(matrix, sparse.data._data_matrix), "Input connection matrix must be sparse!"

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


def input_innervation(sim, base_target=None, neuron_properties=[],
                      t_wins=None):
    from .neuron_groups import load_neurons, load_all_projection_locations
    from .connection_matrix import circuit_cross_group_matrices
    from .neuron_groups.defaults import GID, FIBER_GID

    spikes = input_spikes(sim)
    circ = sim.circuit

    if base_target is None:
        base_target = sim.config.Run["CircuitTarget"]
    nrn = load_neurons(circ, neuron_properties, base_target=base_target)

    nrn = pandas.concat([nrn], keys=["neurons"], names=["__cell_type"])
    nrn = nrn.set_index(nrn.index.droplevel(-1))

    projections = load_all_projection_locations(circ, [])
    projections[GID] = projections[FIBER_GID]
    proj_gids = projections.groupby("projection").apply(lambda x: x[GID].values)

    M = circuit_cross_group_matrices(circ, projections, nrn, connectome=None).stack()
    M = M.droplevel("__cell_type")
    if t_wins is not None:
        innervation = [M.combine(proj_gids,
                                 lambda m, g: input_innervation_from_matrix(spikes, m, g, t_win=t_win))
                       for t_win in t_wins]
        innervation = pandas.concat(innervation, keys=["{0}-{1}".format(*t_win) for t_win in t_wins],
                                    names=["t_win"])
    else:
        innervation = M.combine(proj_gids, lambda m, g: input_innervation_from_matrix(spikes, m, g))
    innervation = innervation.apply(lambda v: pandas.Series(v, index=nrn[GID])).stack()

    return innervation, nrn

