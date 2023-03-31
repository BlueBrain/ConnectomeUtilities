import os

from .defaults import VIRTUAL_FIBERS_FN

SONATA_CONNECTOME_FN = "edges.sonata"
STR_ATLAS_DIR = "atlas_dir"
ATLAS_LOC = "atlas"
PROJECTION_LOC = "projections"
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

def projection_dir(circ):
    circ_base = circuit_base_dir(circ)
    if circ_base is None: return None
    projections = os.path.join(circ_base, PROJECTION_LOC)
    if os.path.exists(projections): return projections
    return None

def projection_fiber_info(circ, projection_name):
    if projection_name in circ.edges:
        projection_fn = circ.edges[projection_name].h5_filepath
        vfib_file = os.path.join(os.path.split(os.path.abspath(projection_fn))[0], VIRTUAL_FIBERS_FN)
        return vfib_file
    
    proj_dir = projection_dir(circ)
    if proj_dir is None: return None
    vfib_file = os.path.join(proj_dir, projection_name, VIRTUAL_FIBERS_FN)
    if os.path.exists(vfib_file): return vfib_file
    return None

def projection_list(circ, return_filename_dict=False):
    import glob
    proj_dir = projection_dir(circ)
    if proj_dir is None: return []

    projections = glob.glob("**/" + SONATA_CONNECTOME_FN, root_dir=proj_dir, recursive=True)
    if return_filename_dict:
        return dict([(os.path.split(_x)[0], os.path.join(proj_dir, _x))
                     for _x in projections])
    projections = [os.path.split(_x)[0] for _x in projections]
    return projections

