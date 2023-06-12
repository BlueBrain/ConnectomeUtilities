# SPDX-License-Identifier: Apache-2.0
import numpy
import pandas
from scipy.spatial.transform import Rotation
from .flatmap_utility import apply_flatmap_with_translation


# The below are just used internally in this package. Not quite redundant with the stuff in neuron_groups.defaults...
COLS_FLAT_XY = ["flat x", "flat y"]
COLS_XYZ = ["x", "y", "z"]


def flat_multi_index(coordinates3d, fm, orientations3d=None):
    coords_flat = apply_flatmap_with_translation(coordinates3d, orientations3d, fm)
    midx = pandas.MultiIndex.from_frame(pandas.DataFrame(coords_flat, columns=COLS_FLAT_XY))
    return midx


def flat_coordinate_frame(coordinates3d, fm, orientations3d=None, grouped=False):
    midx = flat_multi_index(coordinates3d, fm, orientations3d=orientations3d)
    coord_frame = pandas.DataFrame(coordinates3d, index=midx, columns=COLS_XYZ)
    if grouped:
        return coord_frame.groupby(COLS_FLAT_XY).apply(lambda x: x.values)
    return coord_frame


def pandas_flat_coordinate_frame(df_in, fm, columns_xyz=COLS_XYZ, columns_uvw=None, grouped=False):
    xyz = df_in[columns_xyz].values
    uvw = None
    if columns_uvw is not None:
        uvw = df_in[columns_uvw].values

    midx = flat_multi_index(xyz, fm, orientations3d=uvw)
    df_out = df_in.set_index(midx)
    if grouped:
        A = df_out[columns_xyz].groupby(COLS_FLAT_XY).apply(lambda x: x.values)
        return A, df_out
    return df_out


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


class GeneralLinearTransform(object):
    def __init__(self, M):
        self._M = M

    def apply(self, other):
        return numpy.dot(self._M, other.transpose()).transpose()

    def inv(self):
        return GeneralLinearTransform(self._M.transpose())

    def expand(self):
        Mout = numpy.array([
            [self._M[0, 0], 0, self._M[0, 1]],
            [0, 1, 0],
            [self._M[1, 0], 0, self._M[1, 1]]
        ])
        return GeneralLinearTransform(Mout) #Rotation.from_matrix(Mout)


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
            return GeneralLinearTransform(numpy.identity(2)), 2.0
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
    return GeneralLinearTransform(M[:2, :2]), res[1]
