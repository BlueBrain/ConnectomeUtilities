# SPDX-License-Identifier: Apache-2.0
import numpy

from voxcell import VoxcellError

FIX_TRANSITION = 1500  # In the SSCx Bio_M the projection fiber positions are defined at the start of the rays,
# which are outside of Sirio's flatmap. They have to be shifted with 1500 (as of 11.2021) to get them into the flatmap.


def colored_points_to_image(flat_coords, cols, extent=None):
    flat_coords = flat_coords.astype(int)
    if extent is None:
        extent = flat_coords.max(axis=0) + 1
    img = numpy.NaN * numpy.ones(tuple(extent) + (3, ))
    for xy, col in zip(flat_coords, cols):
        img[xy[0], xy[1], :] = col
    return img


def _flatmap_extent(fm, subsample=None):
    mx = fm.raw.max(axis=(0, 1, 2))
    if subsample is not None:
        mx = numpy.floor(mx / subsample).astype(int)
    return mx + 1


def _flat_coordinates_of_regions(names_region, fm, hier, ann, make_unique=True, subsample=None):
    reg_ids = set()
    for region in names_region:
        reg_ids.update(hier.find(region, "acronym", with_descendants=True))
    lst_ids = list(reg_ids)
    view3d = numpy.in1d(ann.raw.flat, lst_ids).reshape(ann.raw.shape)
    view2d = fm.raw[view3d]
    view2d = view2d[numpy.all(view2d >= 0, axis=1)]

    if subsample is not None:
        view2d = numpy.floor(view2d / subsample).astype(int)

    if not make_unique:
        return view2d
    mul = view2d[:, 0].max() + 1
    view_comb = view2d[:, 0] + mul * view2d[:, 1]
    unique_comb = numpy.unique(view_comb)
    return numpy.vstack([
        numpy.mod(unique_comb, mul),
        numpy.floor(unique_comb / mul)
    ]).transpose()


def flat_coordinates_of_regions(names_regions, fm, *args, make_unique=False, subsample=None):
    if len(args) == 2:
        return _flat_coordinates_of_regions(names_regions, fm, *args, make_unique=make_unique, subsample=subsample)
    elif len(args) == 1:
        circ = args[0]
        from ..circuit_models.neuron_groups.sonata_extensions import load_atlas_data, load_atlas_hierarchy
        ann = load_atlas_data(circ, "brain_regions")
        hier = load_atlas_hierarchy(circ)
        return _flat_coordinates_of_regions(names_regions, fm, hier, ann, make_unique=make_unique, subsample=subsample)
    else:
        raise ValueError()


def flat_region_image(lst_regions, fm, *args, extent=None, subsample=None):
    if extent is None:
        extent = _flatmap_extent(fm, subsample=subsample)
    xbins = numpy.arange(extent[0] + 1)
    ybins = numpy.arange(extent[1] + 1)
    counters = []
    for reg in lst_regions:
        if not isinstance(reg, list):
            reg = [reg]
        A = flat_coordinates_of_regions(reg, fm, *args, make_unique=False, subsample=subsample)
        H = numpy.histogram2d(A[:, 0], A[:, 1], bins=(xbins, ybins))[0]
        counters.append(H)

    region_lookup = numpy.argmax(numpy.dstack(counters), axis=2)
    region_lookup[numpy.sum(numpy.dstack(counters), axis=2) == 0] = -1
    return region_lookup


def apply_flatmap_with_translation(xyz, uvw, fm, max_translation=2000):
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
    if uvw is not None and not numpy.all(solution > 0):
        # 1)
        solution = fm.lookup(xyz + FIX_TRANSITION * uvw)
        if numpy.all(solution > 0):
            return solution
        else:
            # 2)
            fac = 0
            step = fm.voxel_dimensions[0] / 4
            tl_factors = numpy.zeros((len(uvw), 1))
            solution = fm.lookup(xyz)
            while numpy.any(solution < 0) and fac < max_translation:
                try:
                    fac += step
                    to_update = numpy.any(solution < 0, axis=1)
                    tl_factors[to_update, 0] = fac
                    solution[to_update, :] = fm.lookup(xyz[to_update, :] + tl_factors[to_update, :] * uvw[to_update, :])
                except VoxcellError:
                    break
    return solution
