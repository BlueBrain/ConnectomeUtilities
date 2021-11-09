import numpy
import pandas
from scipy.spatial.transform import Rotation


def flat_coordinate_frame(coordinates3d, fm, grouped=False):
    coords_flat = fm.lookup(coordinates3d.values)
    coord_frame = pandas.DataFrame(coordinates3d, index=pandas.MultiIndex.from_tuples(map(tuple, coords_flat),
                                                                                      names=["f_x", "f_y"]),
                                 columns=["x", "y", "z"])
    if grouped:
        return coord_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return coord_frame


def neuron_flat_coordinate_frame(circ, fm, grouped=False):
    coordinates3d = circ.cells.get(properties=["x", "y", "z"])
    coord_frame = flat_coordinate_frame(coordinates3d, fm)
    coord_frame["gid"] = coordinates3d.index.values
    if grouped:
        A = coord_frame[["x", "y", "z"]].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        B = coord_frame["gid"].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        return A, B
    return coord_frame


def voxel_flat_coordinate_frame(fm, in_voxel_indices=False, grouped=False):
    valid = numpy.all(fm.raw > -1, axis=-1)
    vxl_xyz = numpy.vstack(numpy.nonzero(valid)).transpose()
    vxl_flat = fm.raw[vxl_xyz[:, 0], vxl_xyz[:, 1], vxl_xyz[:, 2]]
    if not in_voxel_indices:
        vxl_xyz = vxl_xyz * fm.voxel_dimensions.reshape((1, -1)) + fm.offset.reshape((1, -1))
    vxl_frame = pandas.DataFrame(vxl_xyz, index=pandas.MultiIndex.from_tuples(map(tuple, vxl_flat),
                                                                              names=["f_x", "f_y"]),
                                 columns=["x", "y", "z"])
    if grouped:
        return vxl_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return vxl_frame


class Translation(object):
    def __init__(self, v):
        self._v = v

    def apply(self, other):
        return other + self._v

    def inv(self):
        return Translation(-self._v)


class Projection(object):
    def __init__(self, idx):
        self._idx = idx

    def apply(self, other):
        return other[:, self._idx]


class TwoDRotation(object):
    def __init__(self, M):
        self._M = M

    def apply(self, other):
        return numpy.dot(self._M, other.transpose()).transpose()

    def inv(self):
        return TwoDRotation(self._M.transpose())

    def expand(self):
        Mout = numpy.array([
            [self._M[0, 0], 0, self._M[0, 1]],
            [0, 1, 0],
            [self._M[1, 0], 0, self._M[1, 1]]
        ])
        return Rotation.from_matrix(Mout)


class Combination(object):
    def __init__(self, one, two):
        self._one = one
        self._two = two

    def apply(self, other):
        return self._two.apply(self._one.apply(other))

    def inv(self):
        return Combination(self._two.inv(), self._one.inv())


def flatmap_pixel_gradient(fm_or_frame):
    from .flatmap_utility import colored_points_to_image
    if not isinstance(fm_or_frame, pandas.DataFrame):
        fm_or_frame = voxel_flat_coordinate_frame(fm_or_frame)
    per_pixel = fm_or_frame.groupby(["f_x", "f_y"])
    per_pixel_center = per_pixel.apply(lambda x: numpy.mean(x.values, axis=0))

    pxl_center_vol = colored_points_to_image(numpy.vstack(per_pixel_center.index.values),
                                             numpy.vstack(per_pixel_center.values))
    # Gradients thereof: When we go one step in the flat space, this is how many steps we going in global coordinates
    dx_dfx, dx_dfy = numpy.gradient(pxl_center_vol[:, :, 0])
    dy_dfx, dy_dfy = numpy.gradient(pxl_center_vol[:, :, 1])
    dz_dfx, dz_dfy = numpy.gradient(pxl_center_vol[:, :, 2])

    dfx = numpy.dstack([dx_dfx, dy_dfx, dz_dfx])
    dfy = numpy.dstack([dx_dfy, dy_dfy, dz_dfy])
    return dfx, dfy


def _find_rotation_(v_x, v_y):
    if numpy.any(numpy.isnan(v_x)):
        if numpy.any(numpy.isnan(v_y)):
            return TwoDRotation(numpy.identity(2)), 2.0
        vv = numpy.hstack([v_y, [[0]]])
        vtgt = numpy.array([[0, 1, 0]])
    elif numpy.any(numpy.isnan(v_y)):
        vv = numpy.hstack([v_x, [[0]]])
        vtgt = numpy.array([[1, 0, 0]])
    else:
        vv = numpy.hstack([numpy.vstack([v_x, v_y]), [[0], [0]]])
        vtgt = numpy.array([[1, 0, 0], [0, 1, 0]])
    vv = vv / numpy.linalg.norm(vv, axis=1, keepdims=True)
    res = Rotation.align_vectors(vtgt, vv)
    M = res[0].as_matrix()
    return TwoDRotation(M[:2, :2]), res[1]
