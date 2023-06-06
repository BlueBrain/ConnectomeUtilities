# SPDX-License-Identifier: Apache-2.0
import io
import h5py
import os
import numpy
import pandas as pd


def write_sparse_matrix_payload(hdf_group, dset_pattern="matrix_{0}"):
    def write_dset(mat):
        dset_name = dset_pattern.format(len(hdf_group.keys()))
        from scipy import sparse
        bio = io.BytesIO()
        sparse.save_npz(bio, mat)
        bio.seek(0)
        matrix_bytes = list(bio.read())
        hdf_group.create_dataset(dset_name, data=matrix_bytes)
        return hdf_group.name + "/" + dset_name
    return write_dset


def write_dense_matrix_payload(hdf_group, dset_pattern="matrix_{0}"):
    def write_dset(mat):
        dset_name = dset_pattern.format(len(hdf_group.keys()))
        hdf_group.create_dataset(dset_name, data=mat)
        return hdf_group.name + "/" + dset_name
    return write_dset


def read_sparse_matrix_payload(hdf_dset):
    from scipy import sparse
    raw_data = bytes(hdf_dset[:].astype(numpy.uint8))
    bio = io.BytesIO(raw_data)
    mat = sparse.load_npz(bio)
    return mat


def read_dense_matrix_payload(hdf_dset):
    return numpy.array(hdf_dset)


READ_WRITE = {
    "sparse": (read_sparse_matrix_payload, write_sparse_matrix_payload),
    "dense": (read_dense_matrix_payload, write_dense_matrix_payload)
}


def write_toc_plus_payload(extracted, to_path, payload_type="sparse", format=None):
    # TODO: Could inspect the contents of extraced and determine the payload_type from it.
    path_hdf_store, group_identifier = to_path
    group_identifier_toc = group_identifier + "/toc"
    group_identifier_mat = group_identifier + "/payload"
    func = READ_WRITE[payload_type][1]

    h5_file = h5py.File(path_hdf_store, "a")
    h5_grp_mat = h5_file.require_group(group_identifier_mat)
    h5_grp_mat.attrs["payload_type"] = payload_type
    toc = extracted.apply(func(h5_grp_mat))
    h5_file.close()

    toc.to_hdf(path_hdf_store, key=group_identifier_toc,
                     mode="a", format=(format or "fixed"))


class LazyMatrix:
    """..."""
    from lazy import lazy

    def __init__(self, path_hdf, path_dset, func):
        """..."""
        self._hdf = path_hdf
        self._dset = path_dset
        self._reader = func

    @lazy
    def matrix(self):
        """..."""
        with h5py.File(self._hdf, 'r') as hdf:
            dset = hdf[self._dset]
            matrix = self._reader(dset)
        return matrix


def read_toc_plus_payload(path):
    path_hdf_store, group_identifier = path
    group_identifier_toc = group_identifier + "/toc"
    group_identifier_mat = group_identifier + "/payload"
    with h5py.File(path_hdf_store, "r") as h5:
        func = READ_WRITE[h5[group_identifier_mat].attrs.get("payload_type", "sparse")]  # Default for bw compatibility

    if not os.path.isfile(path_hdf_store):
        raise RuntimeError(f"Missing HDF data at path {path_hdf_store}\n")

    toc = pd.read_hdf(path_hdf_store, key=group_identifier_toc)

    return toc.apply(lambda dset_path: LazyMatrix(path_hdf_store, dset_path, func))