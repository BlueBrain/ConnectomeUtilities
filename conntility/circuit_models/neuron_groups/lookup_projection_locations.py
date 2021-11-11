"""
Extracting and flatmapping projection locations
Codebase by Michael Reimann, adaptation to this use-case Christoph Pokorny and András Ecker
last update: 11.2021
"""

import os
import warnings
import numpy as np
import pandas as pd
from voxcell import VoxcellError
from scipy.spatial import KDTree
import sklearn.cluster
from flatmap_utility import per_pixel_coordinate_transformation

VIRTUAL_FIBERS_FN = "virtual-fibers.csv"
VIRTUAL_FIBERS_XYZ = ["x", "y", "z"]
CELLS_XYZ = ["x", "y", "z"]
VIRTUAL_FIBERS_UVW = ["u", "v", "w"]
VIRTUAL_FIBERS_MASK = "apron"
VIRTUAL_FIBERS_GIDS = "sgid"
FIX_TRANSITION = 1500  # In the SSCx Bio_M the projection fiber positions are defined at the start of the rays,
# which are outside of Sirio's flatmap. They have to be shifted with 1500 (as of 11.2021) to get them into the flatmap.


def projection_locations_3d(projection):
    """Reads .csv file saved in the same directory as the sonata file of the projection
    to get locations and directions of the virtual fibers"""
    vfib_file = os.path.join(os.path.split(os.path.abspath(projection))[0], VIRTUAL_FIBERS_FN)
    if not os.path.isfile(vfib_file):
        raise RuntimeError("Cannot find virtual fiber info for the selected projection!")
    vfib = pd.read_csv(vfib_file)
    if VIRTUAL_FIBERS_MASK is not None:
        vfib = vfib[~vfib[VIRTUAL_FIBERS_MASK]]
    return vfib[VIRTUAL_FIBERS_GIDS].values, vfib[VIRTUAL_FIBERS_XYZ].values, vfib[VIRTUAL_FIBERS_UVW].values


def apply_flatmap(xyz, uvw, fm, max_translation=2000):
    """
    Uses Voxcell to look up locations of `xyz` in the flatmap. If locations are outside the valid region of
    the flatmap (which is usually the case for projections as those start at the bottom of L6)
    and `uvw` is provided (i.e. not `None`), then there are 2 possible options:
    1) The invalid locations are translated along the directions given by `uvw` by a hard-coded fix amount, or
    2) The invalid locations are gradually translated along the directions given by `uvw` until they hit
    the valid volume. `max_translation` defines the maximum amplitude of that translation.
    Locations that never hit the valid volume will return a flat location of (-1, -1).
    :param xyz: numpy.array, N x 3: coordinates in 3d space
    :param uvw: numpy.array, N x 3: directions in 3d space. Optional, can be None.
    :param fm: VoxelData: flatmap
    :param max_translation: float.
    :return: Flat locations of the xyz coordinates in the flatmap.
    """
    solution = fm.lookup(xyz)
    if uvw is not None:
        # 1)
        solution = fm.lookup(xyz + FIX_TRANSITION * uvw)
        if np.all(solution > 0):
            return solution
        else:
            # 2)
            fac = 0
            step = fm.voxel_dimensions[0] / 4
            tl_factors = np.zeros((len(uvw), 1))
            solution = fm.lookup(xyz)
            while np.any(solution < 0) and fac < max_translation:
                try:
                    fac += step
                    to_update = np.any(solution < 0, axis=1)
                    tl_factors[to_update, 0] = fac
                    solution[to_update, :] = fm.lookup(xyz[to_update, :] + tl_factors[to_update, :] * uvw[to_update, :])
                except VoxcellError:
                    break
    return solution


def mask_results_bb(results, c, mask_name):
    """
    :param results: The unmasked output of `get_projection_locations()`
    :param c: bluepy.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :return: The "results" are masked such that only the parts within the bounding box of `mask_name` are returned.
             If the CircuitConfig has an Atlas field with and a flatmap can be loaded then the bounding box and masking
             is done in the 2D flat space. Otherwise in 3D space.
    """
    res_gids, res2d, res3d, resdir = results
    _, mask2d, mask3d = get_neuron_locations(c, mask_name)
    if mask2d is None:
        valid = (res3d >= mask3d.min(axis=0, keepdims=True)) & (res3d <= mask3d.max(axis=0, keepdims=True))
    else:
        mask2d = mask2d[np.all(mask2d >= 0, axis=1)]
        valid = (res2d >= mask2d.min(axis=0, keepdims=True)) & (res2d <= mask2d.max(axis=0, keepdims=True))
    valid = np.all(valid, axis=1)

    res_gids = res_gids[valid]
    if res2d is not None:
        res2d = res2d[valid]
    if res3d is not None:
        res3d = res3d[valid]
    if resdir is not None:
        resdir = resdir[valid]

    return res_gids, res2d, res3d, resdir


def mask_results_dist(results, circ, mask_name, max_dist=None, dist_factor=2.0):
    """
    :param results: The unmasked output of `get_projection_locations()`
    :param circ: bluepy.Circuit
    :param mask_name: str: Name of a cell target of projection that serves as a mask
    :param max_dist: float: (Optional) Maximal distance from the `mask_name` location that is considered valid.
    If not provided, a value will be estimated using `dist_factor`
    :param dist_factor: float: (Optional, default: 2.0) If `max_dist` is None, this will be used to conduct an estimate.
    :return: The `results` are masked such that only the parts within `max_dist` of locations associated with
             `mask_name` are returned. If the CircuitConfig has an Atlas field with and a flatmap can be loaded then
             the bounding box and masking is done in the 2D flat space. Otherwise in 3D space.
    """
    res_gids, res2d, res3d, resdir = results
    _, mask2d, mask3d = get_neuron_locations(circ, mask_name)

    if mask2d is None:
        use_res = res3d
        use_mask = mask3d
    else:
        use_res = res2d
        use_mask = mask2d

    t_res = KDTree(use_res)
    t_mask = KDTree(use_mask)
    if max_dist is None:
        dists, _ = t_res.query(use_res, 2)
        max_dist = dist_factor * dists[:, 1].mean()
    actives = t_mask.query_ball_tree(t_res, max_dist)
    actives = np.unique(np.hstack(actives).astype(int))

    res_gids = res_gids[actives]
    if res2d is not None:
        res2d = res2d[actives]
    if res3d is not None:
        res3d = res3d[actives]
    if resdir is not None:
        resdir = resdir[actives]

    return res_gids, res2d, res3d, resdir


def flat_coordinate_frame(pos3d, dir3d, fm, grouped=False):
    """Return same format as flatmap_utility.py/flat_coordinate_frame() but using the local `apply_flatmap()` fn.
    developed for the projections instead of the vanilla `VoxcellData.lookup()`"""
    coords_flat = apply_flatmap(pos3d, dir3d, fm)
    coord_frame = pd.DataFrame(pos3d, columns=["x", "y", "z"],
                               index=(pd.MultiIndex.from_tuples(map(tuple, coords_flat), names=["f_x", "f_y"])))
    if grouped:
        return coord_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return coord_frame


def projection_flat_coordinate_frame(gids, pos3d, dir3d, fm, grouped=False):
    """Return same format as flatmap_utility.py/neuron_flat_coordinate_frame() but takes
    precomputed `gids` and `pos3d` as input not a `bluepy.Circuit`"""
    coord_frame = flat_coordinate_frame(pos3d, dir3d, fm)
    coord_frame["gid"] = gids
    if grouped:
        A = coord_frame[["x", "y", "z"]].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        B = coord_frame["gid"].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        return A, B
    return coord_frame


def supersampled_projection_locations(gids, pos3d, dir3d, fm, orient, pixel_sz=34.0):
    """Function based on flatmap_utility.py/supersampled_neuron_locations()"""
    proj_loc_frame, proj_gid_frame = projection_flat_coordinate_frame(gids, pos3d, dir3d, fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    idxx = proj_loc_frame.index.intersection(tf.index)

    res = tf[idxx].combine(proj_loc_frame[idxx], lambda a, b: a.apply(b))
    final = res.index.to_series().combine(res, lambda a, b: np.array(a) * pixel_sz + b)
    final_frame = np.vstack(final.values)
    out = pd.DataFrame(final_frame, columns=["flat x", "flat y"],
                       index=pd.Index(np.hstack(proj_gid_frame[idxx].values), name="gid"))
    return out


def get_projection_locations(c, projection_name, mask=None, mask_type="bbox", supersample=False):
    """Gets gids, 2D locations in flat space (if flatmap is available), 3D locations and directions of projections
    masked with a given target region defined in the circuit."""
    # get projection locations
    circ_proj = c.config["projections"]
    if projection_name in circ_proj:
        gids, pos3d, dir3d = projection_locations_3d(circ_proj[projection_name])
    else:
        raise RuntimeError("Projection: %s is not part of the CircuitConfig" % projection_name)
    # apply flatmap
    atlas = "atlas" in list(c.config.keys())
    if atlas:
        fm = c.atlas.load_data("flatmap")
        pos2d = apply_flatmap(pos3d, dir3d, fm)
    else:
        warnings.warn("No atlas found in the CircuitConfig, so 2D locations won't be returned.")
        if mask is not None:
            warnings.warn("This will seriously affect masking as the 3D locations of the projection fibers are"
                          "(below L6, thes) outside of the circuit's volume")
        pos2d = None
    # mask with region (hopefully) in flat space
    if mask is not None:
        if mask_type == "bbox":
            gids, pos2d, pos3d, dir3d = mask_results_bb((gids, pos2d, pos3d, dir3d), c, mask)
        elif mask_type.find("dist") == 0:
            mask_spec = mask_type.replace("dist", "")  # Extract distance factor, e.g. mask_type = "dist2.0"
            dist_factor = float(mask_spec) if len(mask_spec) > 0 else None
            gids, pos2d, pos3d, dir3d = mask_results_dist((gids, pos2d, pos3d, dir3d), c, mask, dist_factor)
        else:
            raise RuntimeError("Mask type %s unknown!" % mask_type)
    # supersample (only the masked parts)
    if supersample:
        if atlas:
            fm = c.atlas.load_data("flatmap")
            orient = c.atlas.load_data("orientation")
        else:
            raise RuntimeError("Please add Atlas to the CircuitConfig as it's used to load the flatmap and orientation!")
        super_pos_frame = supersampled_projection_locations(gids, pos3d, dir3d, fm, orient)
        idx = np.argsort(super_pos_frame.index.to_numpy())
        pos2d = super_pos_frame.values[idx]

    return gids, pos2d, pos3d, dir3d
