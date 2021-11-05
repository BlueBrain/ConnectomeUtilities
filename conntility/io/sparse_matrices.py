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


def read_sparse_matrix_payload(hdf_dset):
    from scipy import sparse
    raw_data = bytes(hdf_dset[:].astype(numpy.uint8))
    bio = io.BytesIO(raw_data)
    mat = sparse.load_npz(bio)
    return mat


def write_toc_plus_payload(extracted, to_path, format=None):
    path_hdf_store, group_identifier = to_path
    group_identifier_toc = group_identifier + "/toc"
    group_identifier_mat = group_identifier + "/payload"

    h5_file = h5py.File(path_hdf_store, "a")
    h5_grp_mat = h5_file.require_group(group_identifier_mat)
    toc = extracted.apply(write_sparse_matrix_payload(h5_grp_mat))
    h5_file.close()

    toc.to_hdf(path_hdf_store, key=group_identifier_toc,
                     mode="a", format=(format or "fixed"))


class LazyMatrix:
    """..."""
    from lazy import lazy
    def __init__(self, path_hdf, path_dset):
        """..."""
        self._hdf = path_hdf
        self._dset = path_dset

    @lazy
    def matrix(self):
        """..."""
        with h5py.File(self._hdf, 'r') as hdf:
            dset = hdf[self._dset]
            matrix = read_sparse_matrix_payload(dset)
        return matrix


def read_toc_plus_payload(path, for_step):
    path_hdf_store, group_identifier = path
    group_identifier_toc = group_identifier + "/toc"

    if not os.path.isfile(path_hdf_store):
        raise RuntimeError(f"Missing HDF data for step {for_step} at path {path_hdf_store}\n"
                           f"Run {for_step} step with config that sets outputs to HDF first.")

    toc = pd.read_hdf(path_hdf_store, key=group_identifier_toc)

    return toc.apply(lambda dset_path: LazyMatrix(path_hdf_store, dset_path))