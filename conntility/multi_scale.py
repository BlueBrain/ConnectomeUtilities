# SPDX-License-Identifier: Apache-2.0
import numpy
import pandas

from scipy import sparse
from scipy.spatial import KDTree

from .circuit_models.neuron_groups import group_by_grid
from .circuit_models.neuron_groups import load_filter
from .circuit_models.neuron_groups.defaults import GID
from .circuit_models import circuit_connection_matrix, full_connection_matrix


COLS_NAN = ["ss_flat_x", "ss_flat_y"]


def multi_scale_grouping(nrn, radii, properties=["ss_flat_x", "ss_flat_y"]):
    radii = sorted(radii)
    nrn = nrn.reset_index()
    for i, radius in enumerate(radii):
        nrn = group_by_grid(nrn, properties, radius=radius,
                            prefix="level{0}-".format(i + 1)).reset_index()
    #nrn = nrn.rename(columns=dict(zip(properties, ["level0-x", "level0-y"])))
    #nrn["level0-subtarget"] = numpy.arange(len(nrn))

    return nrn


def count_blocks_of_sparse_matrix(M, szs):
    L = len(szs)
    splts = numpy.hstack([0, numpy.cumsum(szs)])
    M = M.tocsc()
    shape = M.shape
    M = sparse.csc_matrix((M.data, M.indices, M.indptr[splts]), shape=(shape[0], L))
    M = M.tocsr()
    M = sparse.csr_matrix((M.data, M.indices, M.indptr[splts]), shape=(L, L))
    return M


class MultiScaleConnectome(object):

    def __init__(self, extent, children, **kwargs):
        self._extent = extent
        self._children = children
        self._props = kwargs
    
    def skip_and_collapse(self, must_be_balanced=True):
        if not self.isleaf:
            if numpy.any([child.isleaf for child in self._children]):
                if must_be_balanced:
                    assert numpy.all([child.isleaf for child in self._children])
                self._children = self.idx
            else:
                grandchildren = []
                for child in self._children:
                    grandchildren.extend(child._children)
                self._children = grandchildren
                [child.skip_and_collapse(must_be_balanced=must_be_balanced)
                for child in self._children]

    @property
    def isleaf(self):
        return isinstance(self._children, numpy.ndarray)
    
    @property
    def idx(self):
        if self.isleaf:
            return self._children
        return numpy.hstack([child.idx for child in self._children])
    
    @property
    def depth(self):
        if self.isleaf:
            return 0
        return numpy.max([x.depth for x in self._children]) + 1
    
    def count(self, at_reach=None):
        if at_reach == 0:
            return 1
        if self.isleaf:
            return len(self._children)
        if at_reach is None:
            return numpy.sum([x.count() for x in self._children])
        return numpy.sum([x.count(at_reach=at_reach - 1) for x in self._children])
    
    def evaluate_at_depth(self, func, depth):
        selfdepth = self.depth
        assert selfdepth >= depth and depth >= 0
        if selfdepth == depth:
            return [func(self)]
        ret = []
        for child in self._children:
            ret.extend(child.evaluate_at_depth(func, depth))
        return ret
    
    def nrn(self):
        if self.isleaf:
            return self._children
        return pandas.concat([x.nrn() for x in self._children])
    
    @staticmethod
    def __nearest_neighbor_interpolation_for_nans__(nrn, cols_use=["x", "y", "z"],
                                                    add_noise=1.0):
        iv = numpy.any(numpy.isnan(nrn[COLS_NAN]), axis=1)
        K = KDTree(nrn[~iv][cols_use])
        D, idx = K.query(nrn[iv][cols_use])
        print("Interpolation of NaN locations: {0} neurons; distance: {1}, {2}, {3} (min/mean/max)".format(len(D), D.min(), D.mean(), D.max()))
        nrn.loc[iv, COLS_NAN] = nrn.loc[~iv, COLS_NAN].iloc[idx].values + numpy.random.rand(len(idx), len(COLS_NAN)) * add_noise - add_noise / 2
        return nrn

    @classmethod
    def from_circuit(cls, circ, load_cfg_or_df, leafsize=100, nan_policy="interpolate"):
        if isinstance(load_cfg_or_df, dict):
            nrn = load_filter(circ, load_cfg_or_df)
        else:
            nrn = load_cfg_or_df
        if nan_policy == "interpolate":
            nrn = cls.__nearest_neighbor_interpolation_for_nans__(nrn)
        elif nan_policy == "drop":
            nrn = nrn.loc[~numpy.any(numpy.isnan(nrn[COLS_NAN]), axis=1)]
        elif numpy.any(numpy.isnan(nrn[COLS_NAN])):
            raise ValueError("Unknown nan_policy: {0}".format(nan_policy))

        nrn = nrn.sort_values("gid")

        data = nrn[["ss_flat_x", "ss_flat_y"]]
        bbox = list(zip(data.min(), data.max()))
        T = KDTree(nrn[["ss_flat_x", "ss_flat_y"]], leafsize=leafsize)
        
        def _recursive(t, bbox, offset=0):
            if isinstance(t, T.innernode):
                bbox_gr = [
                    lim if i != t.split_dim
                    else (t.split, lim[1]) for i, lim in enumerate(bbox)
                ]
                bbox_l = [
                    lim if i != t.split_dim
                    else (lim[0], t.split) for i, lim in enumerate(bbox)
                ]
                children = []
                child, offset_out = _recursive(t.greater, bbox_gr, offset=offset)
                children.append(child)
                child, offset_out = _recursive(t.less, bbox_l, offset=offset_out)
                children.append(child)
                return cls(bbox, children, offset=offset), offset_out
            return cls(bbox, t.idx, offset=offset), offset + len(t.idx)
        ret, _ = _recursive(T.tree, bbox)
        ret._props["neurons"] = nrn
        ret._props["circuit"] = circ
        return ret
    
    def __attach_matrices__(self, M, tgt_range=10):
        n_node = numpy.mean(self.evaluate_at_depth(lambda x: x.count(), 0))
        r_node = int(numpy.floor(numpy.log2(n_node)))

        idx = self.idx
        nrn_df = self._props["neurons"][["ss_flat_x", "ss_flat_y"]]
        M = M[:, idx][idx]  # Now the rows/cols are in the order given by the multi-scale structure

        def assign_neuron_resolution_mat(mat, blck_lens, reinitialize=False):
            blck_off = numpy.hstack([0, numpy.cumsum(blck_lens)]).tolist()
            assert mat.shape[0] == blck_off[-1]
            def out_func(node):
                fr = blck_off[0]; to = blck_off[1]
                blck_off.pop(0)
                node._props["matrix"] = mat[numpy.ix_(range(fr, to), range(fr, to))]
                if reinitialize:
                    node._props["matrix"] = node._props["matrix"].tocoo().tocsr()
            return out_func
        
        def node_locations_at_resolution(depth_resolution):
            def out_func(node):
                res = node.evaluate_at_depth(lambda x: list(map(numpy.mean, x._extent)), depth_resolution)
                return numpy.vstack(res)
            return out_func
        
        def assign_results_to(func, name_str):
            def out_func(node):
                node._props[name_str] = func(node)
            return out_func

        # Neuron level resolution
        print("At resolution level: neurons")
        lengths_at_sampling = self.evaluate_at_depth(lambda x: x.count(at_reach=tgt_range), tgt_range - r_node)
        _ = self.evaluate_at_depth(assign_results_to(lambda x: nrn_df.iloc[x.idx].values,
                                                     "node_locations"), tgt_range - r_node)
        _ = self.evaluate_at_depth(assign_neuron_resolution_mat(M, lengths_at_sampling), tgt_range - r_node)

        for curr_res in range(0, self.depth - tgt_range + 1):
            print("At resolution level: {0}".format(curr_res))
            lengths_at_resolution = self.evaluate_at_depth(lambda x: x.count(at_reach=1), curr_res)
            _ = self.evaluate_at_depth(assign_results_to(node_locations_at_resolution(curr_res),
                                                         "node_locations"), tgt_range + curr_res)
            lengths_at_sampling = self.evaluate_at_depth(lambda x: x.count(at_reach=tgt_range), tgt_range + curr_res)
            M = count_blocks_of_sparse_matrix(M, lengths_at_resolution)
            _ = self.evaluate_at_depth(assign_neuron_resolution_mat(M, lengths_at_sampling, reinitialize=True),
                                                                    tgt_range + curr_res)
        
    def __remove_unattached_nodes__(self):
        if not self.isleaf:
            children = []
            array_flag = False
            for child in self._children:
                r = child.__remove_unattached_nodes__()
                array_flag = array_flag or isinstance(r, numpy.ndarray)
                children.extend(r)
            if array_flag:
                self._children = numpy.array(children)
            else:
                self._children = children

        if "matrix" in self._props:
            return [self]
        return self._children

    def attach_matrices(self, matrix_kwargs={}, tgt_range=10, force_full_matrix=False):
        circ = self._props["circuit"]
        gids = self._props["neurons"][GID]
        if force_full_matrix:
            assert len(gids) == circ.nodes.size

        if not isinstance(matrix_kwargs, list):
            matrix_kwargs = [matrix_kwargs]
        M = sparse.csc_matrix((len(gids), len(gids)), dtype=float)
        for m_kw in matrix_kwargs:
            if force_full_matrix:
                Madd = circuit_connection_matrix(circ, **m_kw)
            else:
                Madd = circuit_connection_matrix(circ, for_gids=gids, **m_kw)
            if isinstance(Madd, dict):
                assert len(Madd) == 1
                Madd = list(Madd.values())[0]
            M = M + Madd
        self.__attach_matrices__(M, tgt_range=tgt_range)
        self.__remove_unattached_nodes__()
    
    def to_h5(self, fn, group="ms_matrix"):
        import h5py
        with h5py.File(fn, "w") as h5_file:
            h5 = h5_file.require_group(group)
            grp_pattern = "node{0}"

            def __recursive__(n, node_index):
                grp = h5.create_group(grp_pattern.format(node_index))
                grp.create_dataset("matrix", data=numpy.array(n._props["matrix"].todense()))
                grp.create_dataset("node_locations", data=n._props["node_locations"])
                grp.attrs["extent_x"] = n._extent[0]
                grp.attrs["extent_y"] = n._extent[1]

                child_ids = []
                if n.isleaf:
                    grp.create_dataset("neuron_indices", data=n._children)
                else:
                    for child in n._children:
                        child_ids.append(node_index + 1)
                        node_index = __recursive__(child, node_index + 1)
                grp.attrs["child_ids"] = child_ids
                
                return node_index
            __recursive__(self, 0)
            root_grp_name = grp_pattern.format(0)
        self._props["neurons"].to_hdf(fn, key=group + "/" + root_grp_name + "/neurons", format="table")
