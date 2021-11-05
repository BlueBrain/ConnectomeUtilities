import pandas

from ._supersample_utility import *
from ._supersample_utility import _find_rotation_


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
    if tgt_tf == (2, 5): return rot2pixel
    if tgt_tf == (5, 2): return rot2pixel.apply(lambda x: x.inv())
    if tgt_tf == (1, 5): return localized2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 1): return rot2pixel.combine(localized2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    if tgt_tf == (0, 5): return global2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 0): return rot2pixel.combine(global2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    raise ValueError("This should never happen!")


def estimate_flatmap_pixel_size(fm, orient):
    raise NotImplementedError()  # TODO: Implement


def supersample_flatmap(fm, orient, pixel_sz=34.0):
    import voxcell
    vxl_frame = voxel_flat_coordinate_frame(fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    subpixel_loc = vxl_frame.combine(tf, lambda a, b: b.apply(a))
    final_loc = subpixel_loc.index.to_series().combine(subpixel_loc,
                                                       lambda a, b: pixel_sz * numpy.array(a) + b)
    final_loc_arr = numpy.vstack(final_loc.values)
    vxl_loc = numpy.vstack(vxl_frame.values)
    out_raw = -numpy.ones_like(fm.raw, dtype=float)
    out_raw[vxl_loc[:, 0], vxl_loc[:, 1], vxl_loc[:, 2]] = final_loc_arr
    return voxcell.VoxelData(out_raw, fm.voxel_dimensions, offset=fm.offset)


def supersampled_neuron_locations(circ, fm=None, orient=None, pixel_sz=34.0):
    if fm is None:
        fm = circ.atlas.load_data("flatmap")
    if orient is None:
        orient = circ.atlas.load_data("orientation")
    if pixel_sz is None:
        pixel_sz = estimate_flatmap_pixel_size(fm, orient)

    nrn_loc_frame, nrn_gid_frame = neuron_flat_coordinate_frame(circ, fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    idxx = nrn_loc_frame.index.intersection(tf.index)

    res = tf[idxx].combine(nrn_loc_frame[idxx], lambda a, b: a.apply(b))
    final = res.index.to_series().combine(res, lambda a, b: numpy.array(a) * pixel_sz + b)
    final_frame = numpy.vstack(final.values)
    out = pandas.DataFrame(final_frame,
                           columns=["flat x", "flat y"],
                           index=numpy.hstack(nrn_gid_frame[idxx].values))
    return out


def supersampled_projection_fiber_locations(circ, fm=None, orient=None, pixel_sz=34.0):
    # TODO: Implement
    raise NotImplementedError()


# TODO: A Supersampler class that builds the requires transformations and exposes a .transform function
