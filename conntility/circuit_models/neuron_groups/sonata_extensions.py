import os, pandas

from .defaults import VIRTUAL_FIBERS_FN

SONATA_CONNECTOME_FN = "edges.sonata"
STR_ATLAS_DIR = "atlas_dir"
ATLAS_LOC = "atlas"
HIERARCHY_FN = "hierarchy.json"
STR_NODE_SETS = "node_sets_file"


def circuit_base_dir(circ):
    if STR_NODE_SETS in circ.config:
        sonata_base_dir = os.path.split(circ.config[STR_NODE_SETS])[0]
        base_dir, sonata = os.path.split(sonata_base_dir)
        if sonata == "sonata":
            return base_dir
        return sonata_base_dir
    return None


def atlas_dir(circ):
    if STR_ATLAS_DIR in circ.config["components"]:
        return circ.config["components"][STR_ATLAS_DIR]
    else:
        circ_base = circuit_base_dir(circ)
        if circ_base is None: return None
        atlas = os.path.join(circ_base, ATLAS_LOC)
        if os.path.exists(atlas): return atlas
    return None

def load_atlas_data(circ, atlas_name):
    import voxcell

    atlas = atlas_dir(circ)
    if atlas is None: raise RuntimeError("No atlas directory found!")
    atlas_fn = os.path.join(atlas, atlas_name)
    if os.path.splitext(atlas_fn)[1] != ".nrrd":
        atlas_fn = atlas_fn + ".nrrd"
    
    return voxcell.VoxelData.load_nrrd(atlas_fn)

def load_atlas_hierarchy(circ):
    import voxcell

    atlas = atlas_dir(circ)
    if atlas is None: raise RuntimeError("No atlas directory found!")
    hier_fn = os.path.join(atlas, HIERARCHY_FN)
    return voxcell.RegionMap.load_json(hier_fn)

def input_spikes(sim):
    """
    Returns a pandas.Series of the input spikes given to a Simulation.
    Input:
    sim (bluepysnap.Simulation)

    Returns:
    spikes, pandas.Series of input spike ids. Formatted the same as Simulation.spikes
    (i.e. the output spikes).
    """
    # TODO: UPDATE FOR SONATA!
    def read_csv(path):
        data = pandas.read_csv(path, delim_whitespace=True)["/scatter"]
        data.name = "gid"
        data.index.name = "t"
        return data

    sim_root = sim._config._config_dir

    stim = [stim for stim in sim.config.typed_sections("Stimulus") if stim["Pattern"] == "SynapseReplay"]
    if len(stim) == 0:
        return pandas.Series([], index=pandas.Float64Index([], name="t"), name="gid", dtype=float)

    stim = [_stim["SpikeFile"] for _stim in stim]
    stim = [_stim if os.path.isabs(_stim) else os.path.join(sim_root, _stim) for _stim in stim]
    spks = pandas.concat([read_csv(_stim) for _stim in stim], axis=0)
    return spks

def simulation_conditions(sim):
    """
    Return simulation conditions of a bluepysnap.Simulation, looked up from
    the associated sim_conditions.json, if it exists
    """
    from os import path
    import json
    
    sim_path = sim._config._config_dir
    sim_json = sim_path + "sim_conditions.json"
    if not path.isfile(sim_json):
        return {}
    with open(sim_json, "r") as fid:
        conds = json.load(fid)
    return conds
