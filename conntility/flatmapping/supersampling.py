# SPDX-License-Identifier: Apache-2.0
import pandas
from ._supersample_utility import *
from ._supersample_utility import _find_rotation_


DEFAULT_COLUMNS = ["x", "y"]
DEFAULT_WITH_DEPTH = ["x", "depth", "y"]
TMP_INDEX = "__index__"


def per_pixel_coordinate_transformation(fm, orient, from_system="global", to_system="rotated"):
    """
    Systems:
    global: The global coordinate system in um or voxel indices.
    localized: The global coordinate system, but origin moved to the pixel center
    rotated: A local coordinate system, origin at the pixel center, y-axis rotated vertical
    rotated_flat: A local coordinate system, origin at the pixel center, y-axis rotated vertical and then flattened away
    through parallel projection
    subpixel: A local coordinate system, origin at the pixel center, y-axis rotated vertical and then flattened away
    through parallel projection, x and z axes oriented like the flat-x and flat-y axes of the flat map.
    subpixel_depth: A local coordinate system, origin at the pixel center, y-axis rotated vertical,
    x and z axes oriented like the flat-x and flat-y axes of the flat map.
    """
    lst_systems = ["global", "localized", "rotated", "rotated_flat", "subpixel", "subpixel_depth"]
    try:
        tgt_tf = (lst_systems.index(from_system), lst_systems.index(to_system))
    except ValueError:
        raise ValueError("from_system and to_system must be in: {0}, but you provided {1}".format(lst_systems,
                                                                                                  (from_system,
                                                                                                   to_system)))
    invalid_combos = [(2, 3), (3, 2), (3, 1), (3, 0), (4, 2), (4, 1), (4, 0), (3, 5), (4, 5), (5, 3), (5, 4)]
    if tgt_tf in invalid_combos:
        raise ValueError("Invalid combination!")
    if tgt_tf[0] == tgt_tf[1]:
        raise ValueError("Identity transformation not supported")

    vxl_frame = voxel_flat_coordinate_frame(fm)
    per_pixel = vxl_frame.groupby(["f_x", "f_y"])

    per_pixel_negative_center = per_pixel.apply(lambda x: -numpy.mean(x.values, axis=0))
    global2localized = per_pixel_negative_center.apply(Translation)
    if tgt_tf == (0, 1): return global2localized
    if tgt_tf == (1, 0): return global2localized.apply(lambda x: x.inv())

    per_pixel_orient = per_pixel_negative_center.apply(lambda x: orient.lookup(-x))
    localized2rotated = per_pixel_orient.apply(lambda o_vec: Rotation.from_quat(numpy.hstack([o_vec[1:],
                                                                                       o_vec[0:1]])).inv())
    if tgt_tf == (1, 2): return localized2rotated
    if tgt_tf == (2, 1): return localized2rotated.apply(lambda x: x.inv())

    global2rotated = global2localized.combine(localized2rotated, Combination)
    if tgt_tf == (0, 2): return global2rotated
    if tgt_tf == (2, 0): return global2rotated.apply(lambda x: x.inv())

    tf_to_local_flat = Projection([0, 2])
    global2rotflat = global2rotated.apply(lambda base_tf: Combination(base_tf, tf_to_local_flat))
    if tgt_tf == (0, 3): return global2rotflat
    if tgt_tf == (3, 0): return global2rotflat.apply(lambda x: x.inv())

    localized2rotflat = localized2rotated.apply(lambda base_tf: Combination(base_tf, tf_to_local_flat))
    if tgt_tf == (1, 3): return localized2rotflat

    dfx, dfy = flatmap_pixel_gradient(vxl_frame)
    dfx_frame = per_pixel_negative_center.index.to_frame().apply(lambda x: dfx[x["f_x"], x["f_y"]].reshape((1, -1)),
                                                            axis=1)
    dfy_frame = per_pixel_negative_center.index.to_frame().apply(lambda x: dfy[x["f_x"], x["f_y"]].reshape((1, -1)),
                                                            axis=1)
    # Above gradient vectors are in "localized" space. Convert to rotated_flat
    dfx_frame = localized2rotflat.combine(dfx_frame, lambda a, b: a.apply(b))
    dfy_frame = localized2rotflat.combine(dfy_frame, lambda a, b: a.apply(b))

    # We now know the directions of neighboring pixel in the rotated_flat systems
    # Figure out a rotation that transforms the direction vectors to the x or y-axes
    rotflat2pixel_err = dfx_frame.combine(dfy_frame, _find_rotation_)
    rotflat2pixel = rotflat2pixel_err.apply(lambda x: x[0])
    err = rotflat2pixel_err.apply(lambda x: x[1])
    print("Rotation errors: min: {0}, median: {1}, mean: {2}, std: {3}, max: {4}".format(
        err.min(), err.median(), err.mean(), err.std(), err.max()
    ))
    if tgt_tf == (3, 4): return rotflat2pixel
    if tgt_tf == (4, 3): return rotflat2pixel.apply(lambda x: x.inv())
    if tgt_tf == (2, 4): return rotflat2pixel.apply(lambda base_tf: Combination(tf_to_local_flat, base_tf))
    if tgt_tf == (1, 4): return localized2rotflat.combine(rotflat2pixel, Combination)
    if tgt_tf == (0, 4): return global2rotflat.combine(rotflat2pixel, Combination)

    rot2pixel = rotflat2pixel.apply(lambda x: x.expand())

    #  Turn the y-location relative to the pixel geometrical center into a proper depth.
    pp_rotated = per_pixel.apply(lambda x: x.values).combine(global2rotated, func=lambda a, b: b.apply(a))
    tl_to_depth = pp_rotated.apply(lambda locs: Translation(numpy.array([0, -numpy.max(locs[:, 1]), 0])))
    rot2pixel = rot2pixel.combine(tl_to_depth, Combination)

    if tgt_tf == (2, 5): return rot2pixel
    if tgt_tf == (5, 2): return rot2pixel.apply(lambda x: x.inv())
    if tgt_tf == (1, 5): return localized2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 1): return rot2pixel.combine(localized2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    if tgt_tf == (0, 5): return global2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 0): return rot2pixel.combine(global2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    raise ValueError("This should never happen!")


def estimate_flatmap_pixel_size(fm, orient):
    raise NotImplementedError()  # TODO: Implement


def supersample_flatmap(fm, orient, pixel_sz=34.0, include_depth=False):
    import voxcell
    to_system = "subpixel"
    shape = fm.raw.shape[:3] + (2,)
    if include_depth:
        to_system = "subpixel_depth"
        shape = fm.raw.shape[:3] + (3,)
    vxl_frame = voxel_flat_coordinate_frame(fm, grouped=True)
    vxl_index_frame = voxel_flat_coordinate_frame(fm, in_voxel_indices=True, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system=to_system)
    subpixel_loc = vxl_frame.combine(tf, lambda a, b: b.apply(a))
    if include_depth:
        final_loc = subpixel_loc.index.to_series().combine(subpixel_loc,
                                                           lambda a, b: pixel_sz * numpy.array([a[0], 0, a[1]]) + b)
    else:
        final_loc = subpixel_loc.index.to_series().combine(subpixel_loc,
                                                           lambda a, b: pixel_sz * numpy.array(a) + b)
    final_loc_arr = numpy.vstack(final_loc.values)
    if include_depth:
        final_loc_arr[:, 1] = final_loc_arr[:, 1] * -1
    vxl_loc = numpy.vstack(vxl_index_frame.values)
    out_raw = -numpy.ones(shape, dtype=float)
    out_raw[vxl_loc[:, 0], vxl_loc[:, 1], vxl_loc[:, 2]] = final_loc_arr
    return voxcell.VoxelData(out_raw, fm.voxel_dimensions, offset=fm.offset)


def supersampled_locations(df_in, columns_xyz, columns_uvw=None, columns_out=None,
                           circ=None, fm=None, orient=None, pixel_sz=34.0,  #TODO: use DEFAULT_PIXEL_SZ
                           column_index=None, include_depth=False):
    from ..circuit_models.neuron_groups.sonata_extensions import load_atlas_data
    if circ is None:
        assert fm is not None and orient is not None, "Must provide circuit or flatmap and orientation atlas!"
    if fm is None:
        fm = load_atlas_data(circ, "flatmap")
    if orient is None:
        orient = load_atlas_data(circ, "orientation")
    if pixel_sz is None:
        pixel_sz = estimate_flatmap_pixel_size(fm, orient)
    if column_index is None:
        index_frame = df_in.index.to_frame(name=TMP_INDEX)
        df_in = pandas.concat([df_in, index_frame], axis=1)
        column_index = TMP_INDEX
    if columns_out is None:
        if include_depth:
            columns_out = DEFAULT_WITH_DEPTH
        else:
            columns_out = DEFAULT_COLUMNS
    to_system = "subpixel"
    if include_depth:
        to_system = "subpixel_depth"

    loc_frame, df_with_midx = pandas_flat_coordinate_frame(df_in, fm,
                                                           columns_xyz=columns_xyz, columns_uvw=columns_uvw,
                                                           grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system=to_system)
    idxx = loc_frame.index.intersection(tf.index)

    res = tf[idxx].combine(loc_frame[idxx], lambda a, b: a.apply(b))
    if include_depth:
        final = res.index.to_series().combine(res, lambda a, b: numpy.array([a[0], 0, a[1]]) * pixel_sz + b)
    else:
        final = res.index.to_series().combine(res, lambda a, b: numpy.array(a) * pixel_sz + b)
    final_frame = numpy.vstack(final.values)
    if include_depth:
        final_frame[:, 1] = final_frame[:, 1] * -1

    index = df_with_midx[column_index][idxx].values
    out = pandas.DataFrame(final_frame,
                           columns=columns_out,
                           index=index)  # TODO: check that this index is in the right order
    return out
