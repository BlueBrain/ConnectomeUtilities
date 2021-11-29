# -*- coding: utf-8 -*-
"""
Class to get, save and load connection matrix and sample submatrices from it
authors: Michael Reimann, András Ecker
last modified: 11.2021
"""

from operator import ne
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from .circuit_models.neuron_groups.defaults import GID
from .circuit_models.connection_matrix import LOCAL_CONNECTOME

_MAT_GLOBAL_INDEX = 0


class _MatrixNodeIndexer(object):
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent._vertex_properties[prop_name]

    def eq(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop == other]
        return self._parent.subpopulation(pop)

    def isin(self, other):
        pop = self._parent._vertex_properties.index.values[np.in1d(self._prop, other)]
        return self._parent.subpopulation(pop)

    def le(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop <= other]
        return self._parent.subpopulation(pop)

    def lt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop < other]
        return self._parent.subpopulation(pop)

    def ge(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop >= other]
        return self._parent.subpopulation(pop)

    def gt(self, other):
        pop = self._parent._vertex_properties.index.values[self._prop > other]
        return self._parent.subpopulation(pop)

    def random_numerical_gids(self, ref, n_bins=50):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids]
        hist, bin_edges = np.histogram(ref_values.values, bins=n_bins)
        bin_edges[-1] += (bin_edges[-1] - bin_edges[-2]) / 1E9
        value_bins = np.digitize(self._prop.values, bins=bin_edges)
        assert len(hist == len(value_bins[1:-1]))  # `digitize` returns values below and above the spec. bin_edges
        sample_gids = []
        for i in range(n_bins):
            idx = np.where(value_bins == i+1)[0]
            assert idx.shape[0] >= hist[i], "Not enough neurons at this depths to sample from"
            sample_gids.extend(np.random.choice(all_gids[idx], hist[i], replace=False).tolist())
        return sample_gids

    def random_numerical(self, ref, n_bins=50):
        return self._parent.subpopulation(self.random_numerical_gids(ref, n_bins))

    def random_categorical_gids(self, ref):
        all_gids = self._prop.index.values
        ref_gids = self._parent.__extract_vertex_ids__(ref)
        assert np.isin(ref_gids, all_gids).all(), "Reference gids are not part of the connectivity matrix"

        ref_values = self._prop[ref_gids].values
        value_lst, counts = np.unique(ref_values, return_counts=True)
        sample_gids = []
        for i, value in enumerate(value_lst):
            idx = np.where(self._prop == value)[0]
            assert idx.shape[0] >= counts[i], "Not enough %s to sample from" % value
            sample_gids.extend(np.random.choice(all_gids[idx], counts[i], replace=False).tolist())
        return sample_gids

    def random_categorical(self, ref):
        return self._parent.subpopulation(self.random_categorical_gids(ref))


class _MatrixEdgeIndexer(object):
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent.edges[prop_name]

    def eq(self, other):
        idxx = self._prop == other
        return self._parent.subedges(idxx)

    def isin(self, other):
        idxx = np.isin(self._prop, other)
        return self._parent.subedges(idxx)

    def le(self, other):
        idxx = self._prop <= other
        return self._parent.subedges(idxx)

    def lt(self, other):
        idxx = self._prop < other
        return self._parent.subedges(idxx)

    def ge(self, other):
        idxx = self._prop >= other
        return self._parent.subedges(idxx)

    def gt(self, other):
        idxx = self._prop > other
        return self._parent.subedges(idxx)

    def full_sweep(self, direction='decreasing'):
        #  For an actual filtration. Take all values and sweep
        raise NotImplementedError()


class ConnectivityMatrix(object):
    """Small utility class to hold a connections matrix and generate submatrices"""
    def __init__(self, *args, vertex_labels=None, vertex_properties=None,
                 edge_properties=None, default_edge_property="data", shape=None):
        """Not too intuitive init - please see `from_bluepy()` below"""
        """Initialization 1: By adjacency matrix"""
        if len(args) == 1 and isinstance(args[0], np.ndarray) or isinstance(args[0], sparse.spmatrix):
            m = args[0]
            assert m.ndim == 2
            if isinstance(args[0], sparse.spmatrix):
                m = m.tocoo()  # Does not copy data if it already is coo
            else:
                m = sparse.coo_matrix(m)
            self._edges = pd.DataFrame({
                'data': m.data
            })
            if shape is None: shape = m.shape
            else: assert shape == m.shape
            self._edge_indices = pd.DataFrame({
                "row": m.row,
                "col": m.col
            })
            # Adding additional edge properties
            if edge_properties is not None:
                for prop_name, prop_mat in edge_properties.items():
                    self.add_edge_property(prop_name, prop_mat)
        else:
            if len(args) >= 2:
                assert len(args[0]) == len(args[1])
                df = pd.DataFrame({
                    "row": args[0],
                    "col": args[1]
                })
            else:
                df = args[0]
            """Initialization 2: By edge-specific DataFrames"""
            assert edge_properties is not None
            edge_properties = pd.DataFrame(edge_properties)  # In case input is dict
            assert len(edge_properties) == len(df)
            self._edge_indices = df

            if shape is None: 
                shape = tuple(df.max(axis=0).values + 1)
            self._edges = edge_properties
            if default_edge_property not in self.edges:
                default_edge_property = edge_properties.columns[0]  # Or exception?

        # In the future: implement the ability to represent connectivity from population A to B.
        # For now only connectivity within one and the same population
        assert shape[0] == shape[1]
        self._shape = shape
        self.__inititalize_vertex_properties__(vertex_labels, vertex_properties)

        self._default_edge = default_edge_property

        self._lookup = self.__make_lookup__()
        #  NOTE: This part implements the .gids and .depth properties
        for colname in self._vertex_properties.columns:
            #  TODO: Check colname against existing properties
            setattr(self, colname, self._vertex_properties[colname].values)

        # TODO: calling it "gids" might be too BlueBrain-specific! Change name?
        self.gids = self._vertex_properties.index.values

    def __len__(self):
        return len(self.gids)

    def add_vertex_property(self, new_label, new_values):
        assert len(new_values) == len(self), "New values size mismatch"
        assert new_label not in self._vertex_properties, "Property {0} already exists!".format(new_label)
        self._vertex_properties[new_label] = new_values
    
    def add_edge_property(self, new_label, new_values):
        if (isinstance(new_values, np.ndarray) and new_values.ndim == 2) or isinstance(new_values, sparse.spmatrix):
            if isinstance(new_values, sparse.spmatrix):
                new_values = new_values.tocoo()
            else:
                new_values = sparse.coo_matrix(new_values)
            # TODO: Reorder data instead of throwing exception
            assert np.all(new_values.row == self._edge_indices["row"]) and np.all(new_values.col == self._edge_indices["col"])
            self._edges[new_label] = new_values.data
        else:
            assert len(new_values) == len(self._edge_indices)
            self._edges[new_label] = new_values
    
    def __inititalize_vertex_properties__(self, vertex_labels, vertex_properties):
        if vertex_properties is None:
            if vertex_labels is None:
                vertex_labels = np.arange(self._shape[0])
            self._vertex_properties = pd.DataFrame({}, index=vertex_labels)
        elif isinstance(vertex_properties, dict):
            if vertex_labels is None:
                vertex_labels = np.arange(self._hape[0])
            self._vertex_properties = pd.DataFrame(vertex_properties, index=vertex_labels)
        elif isinstance(vertex_properties, pd.DataFrame):
            if vertex_labels is not None:
                raise ValueError("""Cannot specify vertex labels separately
                                 when instantiating vertex_properties explicitly""")
            self._vertex_properties = vertex_properties
        else:
            raise ValueError("""When specifying vertex properties it must be a DataFrame or dict""")
        assert len(self._vertex_properties) == self._shape[0]

    def __make_lookup__(self):
        return pd.Series(np.arange(self._shape[0]), index=self._vertex_properties.index)

    def matrix_(self, edge_property=None):
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self.edges[edge_property], (self._edge_indices["row"], self._edge_indices["col"])),
                                 shape=self._shape, copy=False)
    @property
    def edges(self):
        return self._edges

    @property
    def edge_properties(self):
        # TODO: Maybe add 'row' and 'col'?
        return self.edges.columns.values

    @property
    def vertex_properties(self):
        return self._vertex_properties.columns.values
    
    def matrix_(self, edge_property=None):
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self.edges[edge_property], (self._edge_indices["row"], self._edge_indices["col"])),
                                 shape=self._shape)

    @property
    def matrix(self):
        return self.matrix_(self._default_edge)

    def dense_matrix_(self, edge_property=None):
        return self.matrix_(edge_property=edge_property).todense()

    @property
    def dense_matrix(self):
        return self.dense_matrix_()

    def array_(self, edge_property=None):
        return np.array(self.dense_matrix_(edge_property=edge_property))

    @property
    def array(self):
        return self.array_()

    def index(self, prop_name):
        assert prop_name in self._vertex_properties, "vertex property should be in " + str(self.vertex_properties)
        return _MatrixNodeIndexer(self, prop_name)

    def filter(self, prop_name=None):
        if prop_name is None:
            prop_name = self._default_edge
        return _MatrixEdgeIndexer(self, prop_name)

    def default(self, new_default_property, copy=True):
        assert new_default_property in self.edge_properties, "Edge property {0} unknown!".format(new_default_property)
        if not copy:
            self._default_edge = new_default_property
            return self
        return self.__class__(self._edge_indices["row"], self._edge_indices["col"],
                                  edge_properties=self._edges,
                                  vertex_properties=self._vertex_properties, shape=self._shape,
                                  default_edge_property=new_default_property)

    @staticmethod
    def __extract_vertex_ids__(an_obj):
        if hasattr(an_obj, GID):
            return getattr(an_obj, GID)
        return an_obj

    @classmethod
    def from_bluepy(cls, bluepy_obj, load_config=None, gids=None, connectome=LOCAL_CONNECTOME):
        """
        BlueConfig/CircuitConfig based constructor
        :param bluepy_obj: bluepy Simulation or Circuit object
        :param load_config: config dict for loading and filtering neurons from the circuit
        :param gids: array of gids AKA. the nodes of the graph, if not None: the intersection of these gids
                     and the ones loaded based on the `load_config` will be used
        :param connectome: str. that can be "local" which specifies local circuit connectome
                           or the name of a projection to use
        """
        from .circuit_models.neuron_groups import load_filter
        from .circuit_models import circuit_connection_matrix

        if hasattr(bluepy_obj, "circuit"):
            circ = bluepy_obj.circuit
        else:
            circ = bluepy_obj
        
        nrn = load_filter(circ, load_config)
        nrn = nrn.set_index(GID)
        # TODO: decide if this extra filtering is needed (or make load_config optional
        #  and implement gid based property loading in circuit_models.neuron_groups)
        if gids is not None:
            nrn = nrn.loc[nrn.index.intersection(gids)]
        # TODO: think a bit about if it should even be possible to call this for a projection (remove arg. if not...)
        mat = circuit_connection_matrix(circ, for_gids=nrn.index.values, connectome=connectome)
        return cls(mat, vertex_properties=nrn)

    def submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        """Return a submatrix specified by `sub_gids`"""
        m = self.matrix_(edge_property=edge_property).tocsc()
        if sub_gids_post is not None:
            return m[np.ix_(self._lookup[self.__extract_vertex_ids__(sub_gids)],
                            self._lookup[self.__extract_vertex_ids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_vertex_ids__(sub_gids)]
        return m[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        return self.submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, edge_property=None, sub_gids_post=None):
        return np.array(self.dense_submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post))

    def subpopulation(self, subpop_ids):
        """A ConnectivityMatrix object representing the specified subpopulation"""
        assert np.all(np.in1d(subpop_ids, self._vertex_properties.index.values))
        subpop_ids = self.__extract_vertex_ids__(subpop_ids)
        subpop_idx = self._lookup[subpop_ids]
        # TODO: This would be more efficient if the underlying representation was csc.
        subindex = sparse.coo_matrix((range(len(self._edge_indices["row"])),
                                    (self._edge_indices["row"], self._edge_indices["col"])),
                                     copy=False, shape=self._shape).tocsc()
        subindex = subindex[np.ix_(subpop_idx, subpop_idx)].tocoo()

        out_edges = self._edges.iloc[subindex.data]
        out_vertices = self._vertex_properties.loc[subpop_ids]
        return self.__class__(subindex.row, subindex.col, vertex_properties=out_vertices,
                                  edge_properties=out_edges, default_edge_property=self._default_edge,
                                  shape=(len(subpop_ids), len(subpop_ids)))

    def subedges(self, subedge_indices):
        """A ConnectivityMatrix object representing the specified subpopulation"""
        row_indices = self._edge_indices["row"][subedge_indices]
        col_indices = self._edge_indices["col"][subedge_indices]

        if subedge_indices.dtype == bool:
            out_edges = self._edges[subedge_indices]
        else:
            out_edges = self._edges.iloc[subedge_indices]
        return self.__class__(row_indices, col_indices, vertex_properties=self._vertex_properties,
        edge_properties=out_edges, default_edge_property=self._default_edge, shape=self._shape)

    def random_n_gids(self, ref):
        """Randomly samples `ref` number of neurons if `ref` is and int,
        otherwise the same number of neurons as in `ref`"""
        all_gids = self._vertex_properties.index.values
        if hasattr(ref, "__len__"):
            assert np.isin(self.__extract_vertex_ids__(ref),
                           all_gids).all(), "Reference gids are not part of the connectivity matrix"
            n_samples = len(ref)
        elif isinstance(ref, int):  # Just specify the number
            n_samples = ref
        else:
            raise ValueError("random_n_gids() has to be called with an int or something that has len()")
        return np.random.choice(all_gids, n_samples, replace=False)

    def random_n(self, ref):
        return self.subpopulation(self.random_n_gids(ref))

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        full_prefix = prefix + "/" + group_name
        vertex_properties = pd.read_hdf(fn, full_prefix + "/vertex_properties", format="table")
        edges = pd.read_hdf(fn, full_prefix + "/edges")
        edge_idx = pd.read_hdf(fn, full_prefix + "/edge_indices")

        with h5py.File(fn, 'r') as h5:
            data_grp = h5[full_prefix]
            shape = tuple(data_grp.attrs["NEUROTOP_SHAPE"])
            def_edge = data_grp.attrs["NEUROTOP_DEFAULT_EDGE"]
        return cls(edge_idx["row"], edge_idx["col"], vertex_properties=vertex_properties, edge_properties=edges,
                   default_edge_property=def_edge, shape=shape)

    def to_h5(self, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        full_prefix = prefix + "/" + group_name
        self._vertex_properties.to_hdf(fn, key=full_prefix + "/vertex_properties", format="table")
        self._edges.to_hdf(fn, key=full_prefix + "/edges")
        self._edge_indices.to_hdf(fn, key=full_prefix + "/edge_indices")

        with h5py.File(fn, "a") as h5:
            data_grp = h5[full_prefix]
            data_grp.attrs["NEUROTOP_SHAPE"] = self._shape
            data_grp.attrs["NEUROTOP_DEFAULT_EDGE"] = self._default_edge
            data_grp.attrs["NEUROTOP_CLASS"] = "ConnectivityMatrix"


class TimeDependentMatrix(ConnectivityMatrix):
    def __init__(self, *args, vertex_labels=None, vertex_properties=None, edge_properties=None, default_edge_property=None, shape=None):
        if len(args) == 1 and isinstance(args[0], np.ndarray) or isinstance(args[0], sparse.spmatrix):
            raise ValueError("TimeDependentMatrix can only be initialized by edge indices and edge properties")
        if isinstance(edge_properties, dict):
            assert np.all([isinstance(x.columns, pd.Float64Index) for x in edge_properties.values()]),\
                 "Index of edge properties must be a Float64Index"
            edge_properties = pd.concat(edge_properties.values(), keys=edge_properties.keys(), names=["name"], axis=1)
            edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
        else:
            assert isinstance(edge_properties, pd.DataFrame)
            if isinstance(edge_properties.columns, pd.MultiIndex):
                assert len(edge_properties.columns.levels) == 2, "Columns must index time and name"
                if not isinstance(edge_properties.columns.levels[0], pd.Float64Index):
                    assert isinstance(edge_properties.columns.levels[1], pd.Float64Index),\
                        "Time index must be of type Float64Index"
                    edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
                else:
                    assert isinstance(edge_properties.columns.levels[0], pd.Float64Index),\
                        "Time index must be of type Float64Index"
            else:
                assert isinstance(edge_properties.columns, pd.Float64Index),\
                        "Time index must be of type Float64Index"
                edge_properties = pd.concat([edge_properties], axis=1, copy=False, keys=["data"], names=["name"])
                edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
        if default_edge_property is None:
            default_edge_property = edge_properties.columns.levels[1][0]
        self._time = edge_properties.columns.levels[0].min()
        super().__init__(*args, vertex_labels=vertex_labels, vertex_properties=vertex_properties, edge_properties=edge_properties,
                         default_edge_property=default_edge_property, shape=shape)
    
    @property
    def edges(self):
        return self._edges[self._time]
    
    def at_time(self, new_time):
        # TODO: Add a copy=True kwarg that acts like in .default
        if new_time not in self._edges:
            raise ValueError("No time point at {0} given".format(new_time))  # TODO: interpolate to nearest point?
        self._time = new_time
        return self
    
    def add_edge_property(self, new_label, new_values):
        # TODO: Implement
        raise NotImplementedError("Not yet implemented for TimeDependentMatrix")
    
    def default(self, new_default_property, copy=True):
        ret = super().default(new_default_property, copy=copy)
        if copy: ret._time = self._time
        return ret

        
class ConnectivityGroup(object):
    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0].index, pd.MultiIndex)
            self._mats = args[0]
        elif len(args) == 2:
            self._mats = pd.Series(args[1], index=pd.MultiIndex.from_frame(args[0]))
        self._vertex_properties = pd.concat([x._vertex_properties for x in self._mats],
                                            copy=False, axis=0).drop_duplicates()
        
        for colname in self._vertex_properties.columns:
            #  TODO: Check colname against existing properties
            setattr(self, colname, self._vertex_properties[colname].values)

        # TODO: calling it "gids" might be too BlueBrain-specific! Change name?
        self.gids = self._vertex_properties.index.values
    
    @property
    def index(self):
        return self._mats.index

    @staticmethod
    def __loaditem__(args):
        return ConnectivityMatrix.from_h5(*args)
    
    def __load_if_needed__(self, args):
        if isinstance(args, ConnectivityMatrix) or isinstance(args, pd.Series):
            return args
        return self.__loaditem__(args)
    
    def __getitem__(self, key):
        return self.__load_if_needed__(self._mats[key])
    
    @classmethod
    def from_bluepy(cls, bluepy_obj, load_config=None, connectome=LOCAL_CONNECTOME):
        """
        BlueConfig/CircuitConfig based constructor
        :param bluepy_obj: bluepy Simulation or Circuit object
        :param load_config: config dict for loading and filtering neurons from the circuit
        :param connectome: str. that can be "local" which specifies local circuit connectome
                           or the name of a projection to use
        """
        from .circuit_models.neuron_groups import load_group_filter
        from .circuit_models import circuit_group_matrices

        if hasattr(bluepy_obj, "circuit"):
            circ = bluepy_obj.circuit
        else:
            circ = bluepy_obj
        
        nrn = load_group_filter(circ, load_config)

        # TODO: think a bit about if it should even be possible to call this for a projection (remove arg. if not...)
        mats = circuit_group_matrices(circ, nrn, connectome=connectome)
        nrns = [nrn.loc[x].set_index(GID) for x in mats.keys()]
        con_obj = [ConnectivityMatrix(mat, vertex_properties=n) for n, mat in zip(nrns, mats)]
        return cls(pd.Series(con_obj, index=mats.index))
    
    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        raise NotImplementedError()

    def to_h5(self, fn, group_name=None, prefix=None):
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "conn_group"
        full_prefix = prefix + "/" + group_name
        self._vertex_properties.to_hdf(fn, key=full_prefix + "/vertex_properties", format="table")

        matrix_prefix = full_prefix + "/matrices"
        def _store(mat):
            global _MAT_GLOBAL_INDEX
            grp_name = "matrix{0}".format(_MAT_GLOBAL_INDEX)
            mat.to_h5(fn, group_name=grp_name, prefix=matrix_prefix)
            _MAT_GLOBAL_INDEX = _MAT_GLOBAL_INDEX + 1
            return "::".join([fn, matrix_prefix, grp_name])

        mats = self._mats.apply(_store)
        mats.reset_index().to_hdf(fn, key=full_prefix + "/table", format="table")

        with h5py.File(fn, "a") as h5:
            data_grp = h5[full_prefix]
            data_grp.attrs["NEUROTOP_CLASS"] = "ConnectivityGroup"
