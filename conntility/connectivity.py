# SPDX-License-Identifier: Apache-2.0
"""
Classes to get, save and load (static or time dependent) connection matrices and sample submatrices from them
authors: Michael Reimann, András Ecker
last modified: 01.2022
"""

import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats

from .circuit_models.neuron_groups.defaults import GID
from .circuit_models.connection_matrix import LOCAL_CONNECTOME

_MAT_GLOBAL_INDEX = 0


class _MatrixNodeIndexer(object):
    """
    A helper class used to sample sub-networks of a ConnectivityMatrix.
    Instantiate using ConnectivityMatrix.index.
    """
    def __init__(self, parent, prop_name):
        self._parent = parent
        self._prop = parent._vertex_properties[prop_name]
        if isinstance(self._prop.dtype, int) or isinstance(self._prop.dtype, float):
            self.random = self.random_numerical
        else:
            self.random = self.random_categorical

    def eq(self, other):
        """
        Return subnetwork where the value of the indexed property is equal to the provided reference.
        """
        pop = self._parent._vertex_properties.index.values[self._prop == other]
        return self._parent.subpopulation(pop)

    def isin(self, other):
        """
        Return subnetwork where the value of the indexed property is within the list provided as reference.
        """
        pop = self._parent._vertex_properties.index.values[np.in1d(self._prop, other)]
        return self._parent.subpopulation(pop)

    def le(self, other):
        """
        Return subnetwork where the value of the indexed property is less or equal to the provided reference.
        """
        pop = self._parent._vertex_properties.index.values[self._prop <= other]
        return self._parent.subpopulation(pop)

    def lt(self, other):
        """
        Return subnetwork where the value of the indexed property is less than the provided reference.
        """
        pop = self._parent._vertex_properties.index.values[self._prop < other]
        return self._parent.subpopulation(pop)

    def ge(self, other):
        """
        Return subnetwork where the value of the indexed property is greater or equal to the provided reference.
        """
        pop = self._parent._vertex_properties.index.values[self._prop >= other]
        return self._parent.subpopulation(pop)

    def gt(self, other):
        """
        Return subnetwork where the value of the indexed property is greater than the provided reference.
        """
        pop = self._parent._vertex_properties.index.values[self._prop > other]
        return self._parent.subpopulation(pop)

    def random_numerical_gids(self, ref, n_bins=50):
        """
        Return a random subnetwork where the value of the indexed property matches the subnetwork provided as
        reference. For numerical properties. Values will be binned before matching.
        Returns the gids of the random sample.
        """
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
        """
        Return a random subnetwork where the value of the indexed property matches the subnetwork provided as
        reference. For numerical properties. Values will be binned before matching.
        Returns a ConnectivityMatrix to represent the random sample.
        """
        return self._parent.subpopulation(self.random_numerical_gids(ref, n_bins))

    def random_categorical_gids(self, ref):
        """
        Return a random subnetwork where the value of the indexed property matches the subnetwork provided as
        reference. For categorical properties. 
        Returns the gids of the random sample.
        """
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
        """
        Return a random subnetwork where the value of the indexed property matches the subnetwork provided as
        reference. For categorical properties.
        Returns a ConnectivityMatrix to represent the random sample.
        """
        return self._parent.subpopulation(self.random_categorical_gids(ref))


class _MatrixEdgeIndexer(object):
    """
    A helper class used to sample filtered versions of a network, i.e. same set of nodes, but a subset of edges.
    Instantiate using ConnectivityMatrix.filter.
    """
    def __init__(self, parent, prop_name, side=None):
        # TODO: Enable using an 'edge-associated-node-property' as well.
        self._parent = parent
        self._prop_name = prop_name
        self._side = side
        if prop_name in list(parent.edge_properties):
            self._prop = parent.edges[prop_name]
        elif prop_name in list(parent.vertex_properties):
            self._prop = parent.edge_associated_vertex_properties(prop_name)
            if side is not None:
                self._prop = self._prop[side]
        else:
            raise ValueError("Unknown property: {0}".format(prop_name))
    
    def _reduce_(self, idxx):
        if isinstance(idxx, pd.DataFrame):
            return self._parent.subedges(np.all(idxx, axis=1))
        return self._parent.subedges(idxx.values)

    def eq(self, other):
        """
        Return network with edges where the value of the indexed property is equal to the provided reference.
        """
        idxx = self._prop == other
        return self._reduce_(idxx)

    def isin(self, other):
        """
        Return network with edges where the value of the indexed property is within the list provided as reference.
        """
        idxx = np.isin(self._prop, other)
        return self._reduce_(idxx)

    def le(self, other):
        """
        Return network with edges where the value of the indexed property is less or equal to the provided reference.
        """
        idxx = self._prop <= other
        return self._reduce_(idxx)

    def lt(self, other):
        """
        Return network with edges where the value of the indexed property is less than the provided reference.
        """
        idxx = self._prop < other
        return self._reduce_(idxx)

    def ge(self, other):
        """
        Return network with edges where the value of the indexed property is greater or equal to the provided reference.
        """
        idxx = self._prop >= other
        return self._reduce_(idxx)

    def gt(self, other):
        """
        Return network with edges where the value of the indexed property is greater than the provided reference.
        """
        idxx = self._prop > other
        return self._reduce_(idxx)

    def full_sweep(self, direction='decreasing'):
        """
        Return ConnectivityGroup representing a filtration with successively decreasing or increasing threshold values for
        the indexed property.
        """
        #  For an actual filtration. Take all values and sweep
        raise NotImplementedError()
    
    def random_by_vertex_property_ids(self, ref, n_bins=None, is_edges=False):
        """
        TODO: Instead of specifying a node property here, specify it when instantiating this object.
        Return a random subnetwork with the same nodes but only a subset of the edges. The subset is randomly generated 
        based on a reference. 
        The returned subnetwork will match the reference in terms of the distributions of the specified node property for
        source and target nodes of edges.

        Args:
          ref: A reference to match. Must represent a sub-network of the base network that contains either all nodes and
          a subset of edges, or a subset of nodes and all edges between them.
          In the first case, it is a list of edge ids (indices of ConnectivityMatrix._edges).
          In the second case it is either a ConnectivityMatrix object or a list of the "gid" node property of the base 
          network.

          n_bins: If provided, the node property will be binned as specified. Node property must then be numerical.

          is_edges: If set to True, the reference will be interpreted as edge ids. Otherwise, its nature will be tried
          to be inferred.

        Returns:
          A list of edge ids representing the random subnetwork.
        """
        if isinstance(ref, ConnectivityMatrix):
            assert np.all(np.in1d(ref.gids, self._parent.gids))
        else:
            if is_edges:
                ref = self._parent.subedges(ref)
            else:
                try:
                    ref = self._parent.subpopulation(self._parent.__extract_vertex_ids__(ref))
                    print("Interpreting reference as vertex ids. If that is wrong, set is_edges=True")
                except (AssertionError, IndexError):
                    ref = self._parent.subedges(ref)
                    print("Interpreting reference as edge ids!")

        ref_edges = ref.edge_associated_vertex_properties(self._prop_name)
        if self._side is not None: ref_edges = ref_edges[self._side]
        parent_edges = self._prop #self._parent.edge_associated_vertex_properties(prop_name)

        if n_bins is not None:
            mn, mx = np.min(parent_edges.values.flat), np.max(parent_edges.values.flat)
            bins = np.linspace(mn, mx + (mx - mn) / 1E9, n_bins + 1)
            ref_edges = ref_edges.apply(np.digitize, axis=0, bins=bins)
            parent_edges = parent_edges.apply(np.digitize, axis=0, bins=bins)

        ref_counts = ref_edges.value_counts()
        parent_edges = parent_edges.reset_index().set_index(list(ref_counts.index.names))["index"]

        out_edges = []
        for _idx, n in ref_counts.items():
            out_edges.extend(np.random.choice(parent_edges[_idx].values, n, replace=False))
        return out_edges
    
    def random_by_vertex_property(self, ref, n_bins=None):
        """
        Generates a random subnetwork containing all nodes and a subset of edges. Based on matching
        the distribution of properties of source and target nodes to a reference subnetwork.
        For details, see .random_by_vertex_property_ids

        Returns:
          A ConnectivityMatrix object representing the subnetwork.
        """
        edge_ids = self.random_by_vertex_property_ids(ref, n_bins=n_bins)
        return self._parent.subedges(edge_ids)


class _MatrixNeighborhoodIndexer(object):
    """
    A helper class used to generate subnetworks of neighborhoods of nodes, i.e. containing a specified node
    and all nodes connected to it.
    """

    def __init__(self, parent):
        self._parent = parent
        self._prop = parent._lookup
    
    def get_single(self, pre=None, post=None, center_first=True):
        """
        Get the neighborhood of a single node.

        Args:
          pre: The id of a node in the network. The subnetwork returned will contain that node and all its 
          targets.
          post: The id of a node in the network: The subnetwork returned will contain that node and all its
          sources.
          center_first (optional, default: True): If True, the arguments of "pre" and/or "post" will be 
          listed first in the returned subnetwork. Otherwise, the ordering of the base network is preserved.
        """
        if pre is None and post is None: raise ValueError("Insufficient number of arguments!")

        indexer = self._parent._edge_indices.reset_index()
        idxx = set()
        centers = []
        if pre is not None:
            centers.append(self._prop[pre])
            idxx = idxx.union(indexer.set_index("row")["col"].get(centers[-1:], []))
        if post is not None:
            if pre != post:
                centers.append(self._prop[post])
            idxx = idxx.union(indexer.set_index("col")["row"].get(centers[-1:], []))
        if center_first:
            pop_ids = self._parent._vertex_properties.index[centers + sorted(idxx)]
        else:
            idxx = idxx.union(centers)
            pop_ids = self._parent._vertex_properties.index[sorted(idxx)]
        return self._parent.subpopulation(pop_ids)
        
    def get(self, *args, pre=None, post=None, center_first=True):
        """
        Get the neighborhood of a single or multiple nodes.
        A single non-keyword argument can be provided. If this is done, its value will be used for both 
        the pre= and post= keyword arguments.

        For an explanation of the keyword arguments, see .get_single.
        """
        if len(args) > 1:
            raise ValueError("Please provide a single vertex identifier or use the kwargs!")
        if len(args) == 1:
            arg = args[0]
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                mats = [self.get_single(pre=_arg, post=_arg, center_first=center_first) for _arg in arg]
                df = pd.DataFrame({"center": arg})
                return ConnectivityGroup(df, mats)
            return self.get_single(pre=arg, post=arg, center_first=center_first)
        if not hasattr(pre, "__iter__"):
            if not hasattr(post, "__iter__"):
                return self.get_single(pre, post, center_first=center_first)
            pre = [pre for _ in post]
        if not hasattr(post, "__iter__"): post = [post for _ in pre]
        assert len(pre) == len(post), "Argument mismatch!"
        mats = [self.get_single(_pre, _post, center_first=center_first) for _pre, _post in zip(pre, post)]
        df = pd.DataFrame({"center_pre": pre, "center_post": post})
        return ConnectivityGroup(df, mats)

    def __getitem__(self, idx):
        return self.get(idx)


class ConnectivityMatrix(object):
    """Class to get, save, load and hold a connections matrix and generate submatrices from it"""
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
        # TODO: Additional tests, such as no duplicate edges!
        self.neighborhood = _MatrixNeighborhoodIndexer(self)

    def __len__(self):
        """
        Length of a ConnectivityMatrix is the number of nodes
        """
        return len(self.gids)

    def add_vertex_property(self, new_label, new_values, overwrite=False):
        """
        Assign values for a new property to all nodes. Must be a non-existing property or overwrite=True.

        Args:
          new_label (str): Name of the new property. No property with this name may already exist.
          new_values (iterable): Values for the new property. Must have same length as this object.
          If provided as a DataFrame, must have the same index as obj._vertex_properties
          overwrite (bool, default=False)
        """
        assert len(new_values) == len(self), "New values size mismatch"
        assert overwrite or (new_label not in self._vertex_properties), "Property {0} already exists!".format(new_label)
        self._vertex_properties[new_label] = new_values
    
    def add_edge_property(self, new_label, new_values):
        """
        Assign values for a new property to all edges. Must be a non-existing property.

        Args:
          new_label (str): Name of the new property. No property with this name may already exist.
          new_values (iterable): Values for the new property. Must have length equal to the number of
          edges in this network.
        """
        if (isinstance(new_values, np.ndarray) and new_values.ndim == 2) or isinstance(new_values, sparse.spmatrix):
            if isinstance(new_values, sparse.spmatrix):
                new_values = new_values.tocoo()
            else:
                new_values = sparse.coo_matrix(new_values)
            # TODO: Reorder data instead of throwing exception
            assert np.all(new_values.row == self._edge_indices["row"]) and np.all(new_values.col == self._edge_indices["col"])
            self._edges[new_label] = new_values.data
        else:
            if hasattr(new_values, "values"):
                new_values = new_values.values
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
            raise ValueError("""When specifying vertex properties provide a DataFrame or dict""")
        assert len(self._vertex_properties) == self._shape[0]

    def __make_lookup__(self):
        return pd.Series(np.arange(self._shape[0]), index=self._vertex_properties.index)

    def matrix_(self, edge_property=None):
        """
        Representation of the network as a sparse.coo_matrix.
        Args:
          edge_property (optional): Name of the edge property the values of which to return. If not
          provided, the registered default property is used. (See obj.default())
        """
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self.edges[edge_property], (self._edge_indices["row"], self._edge_indices["col"])),
                                 shape=self._shape, copy=False)
    @property
    def edges(self):
        """
        The list of edges as a DataFrame.
        """
        return self._edges
    
    @property
    def vertices(self):
        """
        The list of nodes contained in this network, along with all their properties.
        """
        return self._vertex_properties.reset_index()

    @property
    def edge_properties(self):
        """
        The names of all available edge properties. Properties can be used with obj.default(prop), obj.matrix_(prop)
        and obj.filter(prop)
        """
        # TODO: Maybe add 'row' and 'col'?
        return self.edges.columns.values

    @property
    def vertex_properties(self):
        """
        The names of all available node properties. Properties can be used with obj.index(prop)
        """
        # TODO: Think about adding GID here as well?
        return self._vertex_properties.columns.values
    
    def edge_associated_vertex_properties(self, prop_name, side=None):
        """
        The list of values of the specified node property associated with all edges. That is, for each edge
        the value of the property is returned for the source and target nodes. 
        Args:
          prop_name: Name of the property to gather. Must be in obj.vertex_properties.

        Returns:
          pandas.DataFrame with the values of the property for source and target node in the columns.
        """
        assert (prop_name in self.vertex_properties) or (prop_name == GID), "{0} is not a vertex property: {1}".format(prop_name, self.vertex_properties)
        eavp = pd.concat(
            [self.vertices[prop_name][self._edge_indices[_idx]].rename(_idx).reset_index(drop=True)
             for _idx in self._edge_indices.columns],
             axis=1, copy=False
        )
        return eavp
    
    def matrix_(self, edge_property=None):
        """
        Representation of the network as a sparse.coo_matrix.
        Args:
          edge_property (optional): Name of the edge property the values of which to return. If not
          provided, the registered default property is used. (See obj.default())
        """
        # TODO: Duplicate?
        if edge_property is None:
            edge_property = self._default_edge
        return sparse.coo_matrix((self.edges[edge_property], (self._edge_indices["row"], self._edge_indices["col"])),
                                 shape=self._shape)

    @property
    def matrix(self):
        """
        Representation of the network as a sparse.coo_matrix. Values associated with each edge are the values of the
        default edge property. See obj.default(prop).
        """
        return self.matrix_(self._default_edge)

    def dense_matrix_(self, edge_property=None):
        """
        Representation of the network as a numpy.matrix. For details, see obj.matrix_()
        """
        return self.matrix_(edge_property=edge_property).todense()

    @property
    def dense_matrix(self):
        """
        Representation of the network as a numpy.matrix. Values associated with each entry are the values of the
        default edge property if an edge exists between the pair, otherwise 0. See obj.default(prop).
        """
        return self.dense_matrix_()

    def array_(self, edge_property=None):
        """
        Representation of the network as a numpy.array. For details, see obj.matrix_()
        """
        return np.array(self.dense_matrix_(edge_property=edge_property))

    @property
    def array(self):
        """
        Representation of the network as a numpy.array. Values associated with each entry are the values of the
        default edge property if an edge exists between the pair, otherwise 0. See obj.default(prop).
        """
        return self.array_()

    def index(self, prop_name):
        """
        Returns an object for the generation of subnetworks based on values of the specified node property.
        For details, see _MatrixNodeIndexer.

        Args:
          prop_name (str): Must be in obj.vertex_properties. Name of the property based on which subnetworks
          are to be sampled. 
        
        Returns:
          _MatrixNodeIndexer associated with this object.
        """
        assert prop_name in self._vertex_properties, "vertex property should be in " + str(self.vertex_properties)
        return _MatrixNodeIndexer(self, prop_name)

    def filter(self, prop_name=None, side=None):
        """
        Returns an object for the generation of subnetworks based on values of the specified edge property.
        For details, see _MatrixEdgeIndexer.

        Args:
          prop_name (str): Must be in obj.edge_properties or obj.vertex_properties.
          Name of the property based on which subnetworks are to be sampled. 
          side (optional, one of "row", "col): Relevant when prop_name is a vertex_property. If "row", then
          the specified vertex property of the source side of the connection is used. If "col", then the
          target side is used. If not provided, the combination of both is used.
        
        Returns:
          _MatrixEdgeIndexer associated with this object.
        """
        if prop_name is None:
            prop_name = self._default_edge
        return _MatrixEdgeIndexer(self, prop_name, side=side)

    def default(self, new_default_property, copy=True):
        """
        Set the default edge property to return. Used in obj.matrix, obj.dense_matrix, obj.array

        Args:
          new_default_property (str): Name of the new edge property to use as default. Must be one of
          obj.edge_properties
          copy (optional, default: True): See below

        Returns:
          If copy is True, returns a copy of this network with the default property set. 
          If copy is False, sets the default property of this network, then returns self.
        """
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
    def from_bluepy(cls, bluepy_obj, load_config=None, connectome=LOCAL_CONNECTOME, **kwargs):
        """
        Sonata config based constructor
        :param bluepy_obj: bluepysnap Simulation or Circuit object
        :param load_config: config dict for loading and filtering neurons from the circuit
        :param connectome: string that specifies the name of an EdgeCollection to load. Must
        be in bluepy_obj.edges, or "local". If "local", then a recurrent connectome is 
        heuristically estimated. In that case it is recommended to specify a node_population
        to load in the load_config. The heuristics for guessing a "local" connectome are
        documented in conntility.circuit_models.circuit_connection_matrix.

        """
        from .circuit_models.neuron_groups import load_filter
        from .circuit_models import circuit_connection_matrix
        from .circuit_models.neuron_groups.grouping_config import _read_if_needed
        #TODO: Support lookup using circuit_node_set_matrix!

        if hasattr(bluepy_obj, "circuit"):
            circ = bluepy_obj.circuit
        else:
            circ = bluepy_obj
        load_config = _read_if_needed(load_config)
        
        if connectome != LOCAL_CONNECTOME:
            if circ.edges[connectome].source.name != circ.edges[connectome].target.name:
                if "source" not in load_config or "target" not in load_config:
                    load_config = {"source": load_config.copy(), "target": load_config.copy()}
                nrn_pre = load_filter(circ, load_config["source"],
                                      node_population=circ.edges[connectome].source.name)
                nrn_post = load_filter(circ, load_config["target"],
                                       node_population=circ.edges[connectome].target.name)
                mat = circuit_connection_matrix(circ, for_gids=nrn_pre.index.values,
                                                for_gids_post=nrn_post.index.values,
                                                connectome=connectome, **kwargs)
                if isinstance(mat, dict):
                    mat = dict([(str(k), v.tocoo()) for k, v in mat.items()])
                    edge_prop_df = pd.DataFrame(dict([(k, v.data) for k, v in mat.items()]))
                    mat = mat[list(mat.keys())[0]]
                else:
                    mat = mat.tocoo()
                    edge_prop_df = pd.DataFrame({"data": mat.data})
                    
                edge_idx_df = pd.DataFrame({
                    "row": mat.row, "col": mat.col + mat.shape[0]
                })
                nrn = pd.concat([nrn_pre, nrn_post], axis=0,
                                 keys=["Source", "Target"],
                                 names=["connection"]).droplevel(1).reset_index()
                nrn.index.name="local_ids"
                return cls(edge_idx_df, vertex_properties=nrn, edge_properties=edge_prop_df,
                           shape=(len(nrn), len(nrn)))

            nrn = load_filter(circ, load_config, node_population=circ.edges[connectome].source.name)
        else:
            nodepop = load_config.get("loading", load_config).get("node_population", None)
            nrn = load_filter(circ, load_config, node_population=nodepop)
        
        nrn = nrn.set_index(GID)
        mat = circuit_connection_matrix(circ, for_gids=nrn.index.values, connectome=connectome, **kwargs)
        if isinstance(mat, dict):
            mat = dict([(str(k), v.tocoo()) for k, v in mat.items()])
            edge_prop_df = pd.DataFrame(dict([(k, v.data) for k, v in mat.items()]))
            mat = mat[list(mat.keys())[0]]
        else:
            mat = mat.tocoo()
            edge_prop_df = pd.DataFrame({"data": mat.data})
            
        edge_idx_df = pd.DataFrame({
            "row": mat.row, "col": mat.col
        })

        return cls(edge_idx_df, vertex_properties=nrn, edge_properties=edge_prop_df, shape=(len(nrn), len(nrn)))

    def submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        """
        Return a submatrix specified by `sub_gids`, represented as sparse.coo_matrix.

        Args:
          sub_gids: List of nodes to return. Values must be in self.gids

          edge_property (optional): Name of the edge property the values of which to return. If not
          provided, the registered default property is used. (See obj.default())

          sub_gids_post (optional): If provided, then the submatrix of connection from sub_gids to
          sub_gids_post is returned. Else square matrix of connectivity from sub_gids to sub_gids.
        
        Returns:
          sparse.coo_matrix corresponding to the spacified submatrix.
        """
        m = self.matrix_(edge_property=edge_property).tocsc()
        if sub_gids_post is not None:
            return m[np.ix_(self._lookup[self.__extract_vertex_ids__(sub_gids)],
                            self._lookup[self.__extract_vertex_ids__(sub_gids_post)])]
        idx = self._lookup[self.__extract_vertex_ids__(sub_gids)]
        return m[np.ix_(idx, idx)]

    def dense_submatrix(self, sub_gids, edge_property=None, sub_gids_post=None):
        """
        See obj.submatrix. But this version returns a numpy.matrix object.
        """
        return self.submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post).todense()

    def subarray(self, sub_gids, edge_property=None, sub_gids_post=None):
        """
        See obj.submatrix. But this version returns a numpy.array.
        """
        return np.array(self.dense_submatrix(sub_gids, edge_property=edge_property, sub_gids_post=sub_gids_post))

    def subpopulation(self, subpop_ids):
        """
        Generate a ConnectivityMatrix object representing the specified subpopulation
        
        Args:
          subpop_ids: List of nodes to return. Entries must be in obj.gids
        
        Returns:
          ConnectivityMatrix representing the subpopulation
        """
        subpop_ids = self.__extract_vertex_ids__(subpop_ids)
        assert np.all(np.in1d(subpop_ids, self._vertex_properties.index.values))
        subpop_idx = self._lookup[subpop_ids]
        # TODO: This would be more efficient if the underlying representation was csc.
        subpop_lookup = pd.Series(range(len(subpop_idx)), index=subpop_idx)
        vld = self._edge_indices["row"].isin(subpop_idx) & self._edge_indices["col"].isin(subpop_idx)
        out_edges = self._edges.loc[vld.values]
        out_indices = self._edge_indices.loc[vld.values]
        if len(out_indices) > 0: out_indices = out_indices.apply(lambda _x: subpop_lookup[_x].values, axis=0)
        out_vertices = self._vertex_properties.loc[subpop_ids]
        
        return ConnectivityMatrix(out_indices,
                                  vertex_properties=out_vertices,
                                  edge_properties=out_edges, default_edge_property=self._default_edge,
                                  shape=(len(subpop_ids), len(subpop_ids)))
    
    def slice(self, angle, position, thickness, columns_slice=["x", "z"], column_y="y"):
        """A ConnectivityMatrix object representing the subpopulation given by a slicing operation.
        First, a slice-specific coordinate system is defined, based on existing coordinates in the
        vertex properties of the object and a specified rotation. The population is then sliced in
        the "depth" coordinate of those coordinates at the specified position, with the specified 
        thickness.

        The slice-specific coordinates are as follows: 
          "slice_x" is given by applying the specified rotation angle to the coordinates in
          "columns_slice".
          "slice_y" is given by "column_y".
          "slice_depth" is orthogonal to "slice_x".

        In summary, angle and position can be randomly chosen within reasonable bounds to generate
        different slices from the population. thickness should be chosed in accordance with the 
        in-vitro experiment one is trying to emulate. column_y should be the coordinate one wants 
        to preserve unchanged in the slice, e.g. cortical depth. columns_slice should be the other
        two coordinates. 
        """
        v = self._vertex_properties[columns_slice].values
        m = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
        V = np.dot(v - np.nanmean(v, axis=0, keepdims=True), m) - np.array([[0, position]])
        in_slice = np.abs(V[:, 1]) < (thickness / 2)
        slice_gids = self.gids[in_slice]
        ret_slice = self.subpopulation(slice_gids)

        slice_y = self._vertex_properties[column_y].values[in_slice]
        ret_slice.add_vertex_property("slice_x", V[in_slice, 0])
        ret_slice.add_vertex_property("slice_y", slice_y)
        ret_slice.add_vertex_property("slice_depth", V[in_slice, 1])

        return ret_slice

    def patch_sample(self, n, mv_mn, mv_cv, columns_xy=["x", "y"], avoidance_range=10.,
                     lim_seed=1., lim_neighborhood=3.):
        """
        Generate a random subpopulation based on spatial locations associated with nodes, trying to emulate
        the spatial biases exhibited in patch-clamp experiments in neuroscience in vitro.

        Args:
          n (int): Number of nodes to sample. Must be smaller than the length of this network (number of nodes).

          mv_mn (list of ints): Mean values of a multivariate gaussian expressing sampling preference. 

          mv_cv (numpy.array): Covariance matrix of a multivariate gaussian expressing sampling preference.

          columns_xy (list of str): Names of node properties to use as the spatial locations of nodes.

          avoidance_range (float): Pairs of nodes below this distance will be avoided to be sampled together.

          lim_seed (float): Maximum distance from the center (as a z-score!) of the population to sample the 
          first node from.

          lim_neighborhood (float): Maximum distance from the rest of the sampled neurons (as a z-score!) to
          sample additional nodes from.

        Returns:
          A ConnectivityMatrix representing a random subpopulation of the specified length. It is sampled as 
          follows:
          The values of the node properties in columns_xy are interpreted as spatial locations associated with
          each node. Note that any number of dimensions is supported, even > 2. mv_mn and mv_cv define a 
          multivariate gaussian in that coordinate system. Note that their lengths must be compatible with the
          number of dimensions. Nodes are preferably sampled from locations close to already sampled nodes 
          according to the spatial kernel codified by the gaussian.
          First a "seed" node is sampled randomly from the center of the population (lim_seed gives) the
          maximal distance from the center that the seed can be, as a z-score of spatial locations. Then
          further nodes are sampled one after another according to the spatial bias provided. 
          Additionally, the "avoidance_range" scales down probabilities for nodes at distances between 0 and
          the specified value. 
        """
        nrn_slice = self._vertex_properties[columns_xy]
        avoidance_range = np.maximum(avoidance_range, 1E-4)  # To avoid division by zero
        p_dist = stats.multivariate_normal(mv_mn, mv_cv)

        marginal_sd = np.sqrt(np.diag(mv_cv))
        vv = ((nrn_slice - nrn_slice.mean()) / nrn_slice.std()).abs() < lim_seed
        idxx = np.nonzero(vv.all(axis=1).values)[0]

        seed_nrn = vv.index[np.random.choice(idxx)]
        seed = nrn_slice.loc[seed_nrn]
        
        zscored = ((nrn_slice - seed) / marginal_sd).abs()
        v_neighborhood = (zscored < lim_neighborhood).all(axis=1) & (zscored > 0.).any(axis=1)
        neighborhood = nrn_slice.loc[v_neighborhood]
        neurons = pd.concat([seed], axis=1).transpose()
        
        other_idx = []
        while len(neurons) < n:
            delta = neighborhood.values.reshape((-1, 1, 2)) - neurons.values.reshape((1, -1, 2))
            avoidance = np.minimum(np.sqrt(np.sum(delta ** 2, axis=-1)) / avoidance_range, 1.)
            p_vec = 1E3 * p_dist.pdf(delta).reshape((-1, len(neurons))) * avoidance
            p_vec = np.prod(p_vec, axis=1)
            
            p_vec = p_vec / p_vec.sum()
            new_idx = np.random.choice(len(p_vec), p=p_vec)
            other_idx.append(new_idx) 
            neurons = pd.concat([neurons, neighborhood.iloc[[new_idx]]], axis=0)
        return self.subpopulation(neurons.index)

    def subedges(self, subedge_indices):
        """
        A ConnectivityMatrix object representing the specified subnetwork. Specifically, a subnetwork
        containing all nodes, but only a subset of edges.

        Args:
          subedge_indices (list): List of edge indices to keep

        Returns:
          ConnectivityMatrix representing the specified subnetwork.
        """
        if isinstance(subedge_indices, pd.Series):
            subedge_indices = subedge_indices.values
        rowcol = self._edge_indices.iloc[subedge_indices]
        out_edges = self._edges.iloc[subedge_indices]

        return ConnectivityMatrix(rowcol["row"], rowcol["col"], vertex_properties=self._vertex_properties,
        edge_properties=out_edges, default_edge_property=self._default_edge, shape=self._shape)
    
    def _active_in_transmission_response(self, eavp, spks_row, spks_col, max_delta_t):
        """Returns boolean array with an entry for each edge. True, if it is active in the transmission
        response graph for the given spiking activity.
        """
        _v = np.isin(eavp["row"], spks_row) & np.isin(eavp["col"], spks_col)
        spks_row = spks_row.reset_index().groupby("gid").agg(list)
        spks_col = spks_col.reset_index().groupby("gid").agg(list)

        row_t = spks_row.loc[eavp["row"][_v]].reset_index(drop=True).applymap(lambda x: np.vstack(x))
        col_t = spks_col.loc[eavp["col"][_v]].reset_index(drop=True).applymap(lambda x: np.array([x]))

        def any_match(series, max_delta_t=5.0):
            delta = series.values[1] - series.values[0]
            return np.any((delta > 0) & (delta <= max_delta_t))

        _v[_v] = pd.concat([row_t, col_t], axis=1).aggregate(any_match, axis=1, max_delta_t=max_delta_t)
        return _v

    def transmission_response(self, spks, t_wins, max_delta_t):
        """
        Returns the transmission response matrices for the given spiking activity in specified time
        windows.

        Args:
          spks (pandas.Series): A Series representing spike times in the network. Each row representing
          a single spike. Index must be a FloatIndex indicating the timing of the spike. Values are
          identifiers of the nodes that spike, i.e. each entry must be in obj.gids.

          t_wins(list of list): A list of time windows to calculate the TR graph for. Each entry must be
          a tuple or list of length 2: [t_start, t_end]. Same units as the spike times.

          max_delta_t (float): Maximum delay for a target node spiking to be considered caused by a source
          node spiking. Same unit as the spike times.
        
        Returns:
          A ConnectivityGroup, i.e. a series of ConnectivityMatrix objects, where each represents a sub-
          network of the same nodes, but only a subset of edges. Subset of edges is picked based on the 
          provided spiking activity as follows: An edge is considered active, if its source spiked in a
          given time window and its target spiked within "max_delta_t" of the source spike (even if that
          target spike is outside the time window!). The ConnectivityGroup contains one Matrix per time
          window.
        """
        spks = spks.loc[np.isin(spks, self.vertices[GID])]
        t_wins = np.array(t_wins).reshape((-1, 2))
        
        eavp = self.edge_associated_vertex_properties("gid")
        for t_start, t_end in t_wins:
            spks_row = spks[t_start:t_end]
            spks_col = spks[t_start:(t_end+max_delta_t)]
            v = self._active_in_transmission_response(eavp, spks_row, spks_col, max_delta_t)
            yield self.subedges(v)
    
    def transmission_response_rates(self, spks, t_wins, max_delta_t, show_progress=False,
                                    normalize="mean"):
        """
        The fraction of edges active in a TR graph with the specified parameters.
        Args:
          spks, t_wins, max_delta_t: See obj.transmission_response(...)
          show_progress (optional, default False): Show progress bar while calculating.
          normalize (str): Must be one of ["mean", "sum", "pre", "expected_simple", "expected_strong"]
        
        Returns:
          The number of edges active in the TR graphs, normalized as specified:
            "sum": No normalization.
            "mean": Relative to the number of structurally present edges.
            "pre": Similar to mean, but taking the out-degree of individual source neurons into account.
            "expected_simple": Compared to expectation, based on the overall firing rate.
            "expected_strong": Compared to expectation, based on firing rates and individual degree distributions.
        """
        spks = spks.loc[np.isin(spks, self.vertices[GID])]
        t_wins = np.array(t_wins).reshape((-1, 2))
        assert normalize in ["mean", "sum", "pre", "expected_simple", "expected_strong"], "Unknown normalization: {0}".format(normalize)

        def empty(arg_in):
            for _x in arg_in: yield(_x)
        gen = empty
        if show_progress:
            from tqdm import tqdm
            gen = tqdm
        
        eavp = self.edge_associated_vertex_properties("gid")
        p = []
        for t_start, t_end in gen(t_wins):
            spks_row = spks[t_start:t_end]
            spks_col = spks[t_start:(t_end+max_delta_t)]
            v = self._active_in_transmission_response(eavp, spks_row, spks_col, max_delta_t)
            if normalize == "mean":
                p.append(v.mean())
            elif normalize == "sum":
                p.append(v.sum())
            elif normalize == "pre":
                den = np.isin(eavp["row"], spks_row).sum()
                p.append(v.sum() / den)
            elif normalize == "expected_simple":
                p1 = float(len(spks_row)) / self._shape[0]
                p2 = float(len(spks_col)) / self._shape[1]
                p.append(v.mean() / (p1 * p2))
            elif normalize == "expected_strong":
                den = np.isin(eavp["row"], spks_row).mean()
                dem = np.isin(eavp["col"], spks_col).mean()
                p.append(v.mean() / (den * dem))
        return np.array(p)

    def random_n_gids(self, ref):
        """
        Generate node identifiers for a random subpopulation of specified size.

        Args:
          ref: Determines the size of the random subsample. Can be an int or an object with length attribute.
          
        Returns:
          A numpy.array containing node identifiers of nodes in this network of specified size.  
        """
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
        """
        Generate a random subpopulation of specified size.

        Args:
          ref: Determines the size of the random subsample. Can be an int or an object with length attribute.
          
        Returns:
          A ConnectivityMatrix representing a subnetwork of the specified number of nodes and all edges between them.  
        """
        return self.subpopulation(self.random_n_gids(ref))
    
    def analyze(self, analysis_recipe):
        """
        Analyze this ConnectivityMatrix according to an analysis recipe.

        Args:
          analysis_recipe: A dict, or the path to a .json file containing a dict. For the format, see 
          configuration_files.md
        
        Returns:
          See configuration_files.md
        """
        from .analysis import get_analyses
        analyses = get_analyses(analysis_recipe)
        res = {}
        for analysis in analyses:
            res[analysis._name] = analysis.apply(self)
        return res
    
    def partition(self, by_columns):
        """
        Split this network into subnetworks according to the values of a node property.

        Args:
          by_columns: A list of strings of node properties. All values must be in obj.vertex_properties.

        Returns:
          A ConnectivityGroup defining a partition of this network, where each group contains the nodes associated
          with a unique combination of values of the specified properties.
        """
        if isinstance(by_columns, str): return self.partition([by_columns])
        str_idx = self._vertex_properties.index.name or "index"
        grp = self.vertices.groupby(by_columns).apply(lambda x: self.subpopulation(x[str_idx]))
        if len(by_columns) == 1:
            grp.index = pd.MultiIndex.from_frame(grp.index.to_frame())
        return ConnectivityGroup(grp)
    
    def condense(self, by_columns, str_original="_idxx_in_original"):
        """
        Generate quotient matrices, i.e. connectivity at "reduced resolution".

        Args:
          by_columns: A list of strings of node properties. All values must be in obj.vertex_properties.

          str_original: A string specifying the name of a node property to be used in the returned ConnectivityMatrix.

        Returns:
          A ConnectivityMatrix of the number of connections between groups of nodes defined by the specified list
          of node properties. Each group contains nodes associated with a unique combination of values of the properties.
          The nodes of the return ConnectivityMatrix will be associated with a property of the specified name. The values
          of the property are lists of indices of the contained nodes in the original ConnectivityMatrix.
        """
        # TODO: Sum of values instead number of edges?
        if isinstance(by_columns, str): return self.condense([by_columns], str_original=str_original)
        orig_vtx = self.vertices
        orig_vtx[str_original] = range(len(orig_vtx))
        orig_vtx = orig_vtx.groupby(by_columns)[str_original].apply(list).sort_index()

        edge_table = pd.concat([self.edge_associated_vertex_properties(_use) for _use in by_columns],
                               axis=1, keys=by_columns)
        edge_table.columns = edge_table.columns.reorder_levels([1, 0])
        node_idx = pd.Series(range(len(orig_vtx)), index=orig_vtx.index)

        orig_vtx.index = node_idx.values

        if len(by_columns) == 1:
            edges = pd.DataFrame({
                        "row": node_idx[edge_table["row"][by_columns[0]]].values,
                        "col": node_idx[edge_table["col"][by_columns[0]]].values},
                        index=edge_table.index).value_counts()
        else:
            def lookup(entry):
                return pd.Series({
                    "row": node_idx.__getitem__(tuple(entry["row"])),
                    "col": node_idx.__getitem__(tuple(entry["col"]))
                })
            edges = edge_table.apply(lookup, axis=1).value_counts()
        MC = ConnectivityMatrix(edges.index.to_frame(), edge_properties={"count": edges.values},
                             shape=(len(orig_vtx), len(orig_vtx)), default_edge_property="count",
                             vertex_properties=orig_vtx.sort_index().to_frame())
        return MC
    
    def core_decomposition(self, str_core_label="_core_decomposition"):
        """
        Partition this network according its core decomposition, as defined in sknetwork.
        """
        from sknetwork.topology import CoreDecomposition
        from sknetwork.utils import directed2undirected

        core = CoreDecomposition()
        labels = core.fit_transform(directed2undirected(self.matrix.tocsr()))
        self.add_vertex_property(str_core_label, labels)
        ret = self.partition(str_core_label)
        self._vertex_properties.drop(str_core_label, axis=1, inplace=True)
        return ret
    
    def __modularity_sknetwork__(self, with_respect_to, resolution_param=None):
        try:
            from sknetwork.clustering import get_modularity as modularity
        except ImportError:
            from sknetwork.clustering import modularity
        if isinstance(with_respect_to, str):
            return self.__modularity_sknetwork__([with_respect_to], resolution_param=resolution_param)
        if resolution_param is None: resolution_param = 1.0

        rel_data = self.vertices[with_respect_to]
        idxx = pd.MultiIndex.from_frame(rel_data).unique().sort_values()
        idxx = pd.Series(range(len(idxx)), index=idxx)
        labels = rel_data.apply(lambda x: idxx.__getitem__(tuple(x)), axis=1)
        return modularity(self.matrix.tocsr(), labels.values, resolution=resolution_param)
    
    def modularity(self, with_respect_to, resolution_param=None, implementation="sknetwork"):
        """
        Calculate the modularity of this network according to a given partition of its nodes.

        Args:
          with_respect_to: A list of strings of node properties that define the partition to use. 
          All values must be in obj.vertex_properties. A group of the partition is defined by a unique
          combination of values of the specified properties.

          resolution_param (float): The resolution parameter. See sknetwork.clustering.get_modularity for details.

          implementation (str): Which implementation of modularit to use. Default is "sknetwork", which will
          use sknetwork.clustering.get_modularity. Any other value will use a custom implementation that is
          part of this package.
        
        Returns:
          If implementation is not "sknetwork", returns a pandas.Series of the contributions to modularity of
          each subnetwork. The actual modularity is the sum of this. If "sknetwork", then only a single value is 
          returned.
        """
        if implementation == "sknetwork": return self.__modularity_sknetwork__(with_respect_to, resolution_param)
        if isinstance(with_respect_to, str): return self.modularity([with_respect_to], resolution_param=resolution_param,
                                                                    implementation=implementation)
        if resolution_param is None: resolution_param = 0.0

        edge_table = pd.concat([self.edge_associated_vertex_properties(_use) for _use in with_respect_to],
                               axis=1, keys=with_respect_to)
        edge_table.columns = edge_table.columns.reorder_levels([1, 0])

        in_module = (edge_table["row"] == edge_table["col"]).all(axis=1)
        edge_table[("value", "value")] = self.edges[self._default_edge].values

        sm = edge_table["value", "value"].sum()  # Total sum of weights in network
        # Sum of weights outgoing / incoming from each module
        frac_out = edge_table[["row", "value"]].droplevel(0, axis=1).groupby(with_respect_to)["value"].agg("sum")
        frac_in = edge_table[["col", "value"]].droplevel(0, axis=1).groupby(with_respect_to)["value"].agg("sum")

        expected = (frac_out * frac_in / sm)  # For each module: Expected sum of weights of connections within the module
        # Actual sums of weights within
        real = edge_table[["row", "value"]].droplevel(0, axis=1).loc[in_module]
        real = real.groupby(with_respect_to)["value"].agg("sum")
        mdlrty = real.subtract(expected, fill_value=0) / sm  # Normalized difference between real and expected

        if resolution_param != 0:
            # Number of potential connections in each module (no autapses!)
            if len(with_respect_to) == 1:
                npairs = self.vertices[with_respect_to[0]].value_counts().apply(lambda x: x ** 2 - x).sort_index()
            else:
                npairs = self.vertices[with_respect_to].value_counts().apply(lambda x: x ** 2 - x).sort_index()
            mdlrty = (real.divide(npairs, fill_value=0) ** resolution_param) * mdlrty
        return mdlrty

    @classmethod
    def from_h5(cls, fn, group_name=None, prefix=None):
        """
        Load a ConnectivityMatrix from a propriatory formatted hdf5 file.

        Args:
          fn: Path to the hdf5 file the data is stored in
          group_name (default: full_matrix): Name of the group used within the file
          prefix (default: connectivity): A prefix within the hdf5 file under which the group is found.

        Returns:
          Loaded ConnectivityMatrix.
        """
        if prefix is None:
            prefix = "connectivity"
        if group_name is None:
            group_name = "full_matrix"
        full_prefix = prefix + "/" + group_name
        vertex_properties = pd.read_hdf(fn, full_prefix + "/vertex_properties")
        edges = pd.read_hdf(fn, full_prefix + "/edges")
        edge_idx = pd.read_hdf(fn, full_prefix + "/edge_indices")

        with h5py.File(fn, 'r') as h5:
            data_grp = h5[full_prefix]
            shape = tuple(data_grp.attrs["NEUROTOP_SHAPE"])
            def_edge = data_grp.attrs["NEUROTOP_DEFAULT_EDGE"]
        return cls(edge_idx["row"], edge_idx["col"], vertex_properties=vertex_properties, edge_properties=edges,
                   default_edge_property=def_edge, shape=shape)

    def to_h5(self, fn, group_name=None, prefix=None):
        """
        Save a ConnectivityMatrix into a propriatory formatted hdf5 file.

        Args:
          fn: Path to the hdf5 file to store the data in
          group_name (default: full_matrix): Name of the group within the file to use to store the data
          prefix (default: connectivity): A prefix within the hdf5 file under which the group will be created.

        """
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


def _update_load_config(load_cfg, sim_tgt):
    from .circuit_models.neuron_groups.grouping_config import _read_if_needed
    load_config = _read_if_needed(load_cfg)
    if load_config is None:
        load_config = {"loading": {"base_target": sim_tgt}}
    elif "loading" in load_config:
        if "base_target" not in load_config["loading"]:
            load_config["loading"]["base_target"] = sim_tgt
    else:  # why is this part necessary?
        load_config["base_target"] = sim_tgt
    return load_config


class StructurallyPlasticMatrix(ConnectivityMatrix):
    """
    A version of ConnectivityMatrix for connectivity that changes over time structurally, i.e.
    the presence and absence of edges may change.
    """
    def __init__(self, *args, vertex_labels=None, vertex_properties=None,
                 edge_properties=None, default_edge_property="data", shape=None,
                 edge_off={}, edge_on={}, check_consistency=True):
        super().__init__(*args, vertex_labels=vertex_labels, vertex_properties=vertex_properties,
                         edge_properties=edge_properties, default_edge_property=default_edge_property,
                         shape=shape)
        self._off = self._build_on_off_index(edge_off)
        self._on = self._build_on_off_index(edge_on)
        if check_consistency:
            check = self.is_consistent()
            failure_count = (~check).sum()
            if failure_count > 0:
                raise ValueError("On-off data is inconsistent for {0} edges!".format(failure_count))
    
    @staticmethod
    def _build_on_off_index(struc_in):
        if isinstance(struc_in, dict):
            t = sorted(struc_in.keys())
            if len(struc_in) > 0:
                idx = pd.Index(np.hstack([_t * np.ones_like(struc_in[_t]) for _t in t]), name="t", dtype="int64")
                vals = np.hstack([struc_in[_t] for _t in t])
            else:
                idx = pd.Index([], name="t", dtype="int64")
                vals = []
            return pd.Series(vals, index=idx, name="edge", dtype="int64").sort_index()
        elif isinstance(struc_in, pd.Series):
            idx = pd.Index(struc_in.index, dtype="int64", name="t")
            struc_in.index = idx
            struc_in.name = "edge"
            return struc_in.sort_index()
        else:
            raise ValueError()
    
    def __getitem__(self, idx):
        is_off = self._off.loc[:idx].drop_duplicates(keep="last")
        is_on = self._on.loc[:idx].drop_duplicates(keep="last")
        off_mx = is_off.reset_index().groupby("edge").agg("max") # the last time it's switched off
        on_mx = is_on.reset_index().groupby("edge").agg("max") # the last time its' switched_on

        idxx = pd.Series(0, index=range(len(self._edge_indices)), name="t").to_frame()
        idxx = idxx.subtract(off_mx.subtract(on_mx, fill_value=-1), fill_value=0) >= 0

        return self.subedges(idxx.index[idxx["t"]].values)
    
    def delta(self, idx_fr, idx_to):
        """
        Return the net changes occuring between two time steps.

        Args:
          idx_fr (int): Index of the step from where changes are to be considered.
          idx_to (int): Index of the step up to which changes are to be considered.

        Returns:
          A ConnectivityMatrix encoding the changes occuring between the time steps. That is, if an edge
          is lost between the time steps, then the corresponding entry will be -1, if an edge if gained 
          it will be 1, otherwise 0. 
          obj.delta(n, n) is always all zeros.
        """
        mxx = self._off.index.max() + 1
        is_off = self._off.get(np.arange(idx_fr + 1, idx_to + 1), self._off.iloc[:0]).reset_index().groupby("edge")
        is_on = self._on.get(np.arange(idx_fr + 1, idx_to + 1), self._on.iloc[:0]).reset_index().groupby("edge")

        off_mx = is_off.agg("max")
        on_mx = is_on.agg("max")
        off_mn = is_off.agg("min")
        on_mn = is_on.agg("min")

        off_first = off_mn.subtract(on_mn, fill_value=mxx) < 0
        off_last = off_mx.subtract(on_mx, fill_value=-1) > 0
        delta_sign = -((off_first.astype(int) + off_last) - 1)
        delta_sign = delta_sign["t"][delta_sign["t"] != 0]

        ret = self.subedges(delta_sign.index)
        ret.add_edge_property("delta", delta_sign.values)
        return ret.default("delta", copy=False)
    
    def skip(self, step, copy=True):
        """
        A version of this Matrix where the change occuring in a single time step is ignored.

        Args:
          step: Index of the time step to ignore.

          copy (boolean, default True): If True, returns a copy. Else this object is modified. 
        
        Returns:
          A StructurallyPlasticMatrix where the specified change is skipped (ignored).
        """
        new_off = self._off.drop(step)
        new_on = self._on.drop(step)
        if copy:
            return StructurallyPlasticMatrix(self._edge_indices, vertex_properties=self._vertex_properties,
                                             edge_properties=self._edges, default_edge_property=self._default_edge,
                                             shape=self._shape, edge_off=new_off, edge_on=new_on,
                                             check_consistency=False).fix_consistency(copy=False)
        self._off = new_off
        self._on = new_on
        return self.fix_consistency(copy=False)
    
    def count_changes(self, count_off=True, count_on=True):
        """
        Counts how often all edges are changing, i.e. lost or gained over all time steps.

        Args:
          count_off (boolean, default: True): If true, count changes where the edge is lost

          count_on(boolean, default: True): If true, count changes where the edge is gained

        Returns:
          A ConnectivityMatrix where each edge of this network is associated with the number
          of times it is gained / lost.
        """
        counts = pd.DataFrame({"count": np.zeros(len(self._edge_indices), dtype=int)})
        if count_off:
            to_add = self._off.reset_index().groupby("edge").agg(len)
            counts = counts.add(to_add.rename(columns={"t": "count"}), fill_value=0)
        if count_on:
            to_add = self._on.reset_index().groupby("edge").agg(len)
            counts = counts.add(to_add.rename(columns={"t": "count"}), fill_value=0)
        counts["data"] = np.ones(len(self._edge_indices), dtype=bool)
        return ConnectivityMatrix(self._edge_indices, vertex_properties=self._vertex_properties,
                                  edge_properties=counts, default_edge_property="count", shape=self._shape)
    
    def amount_active(self):
        """
        Count the number of time steps each edge is active in

        Returns:
          A ConnectivityMatrix where each edge of this network is associated with the number of
          time steps it is active in.
        """
        mxx = np.maximum(self._off.index.max(), self._on.index.max()) + 1
        counts = pd.DataFrame({"count": mxx * np.ones(len(self._edge_indices), dtype=int),
                               "data": np.ones(len(self._edge_indices), dtype=bool)})
        off_times = self._off.reset_index().groupby("edge").apply(lambda x: np.hstack([x["t"].values, mxx]))
        on_times = self._on.reset_index().groupby("edge").apply(lambda x: np.hstack([0, x["t"].values]))

        def counter(arg):
            if not isinstance(arg["ton"], np.ndarray): return arg["toff"][0]
            ton = arg["ton"]
            return (arg["toff"][:len(ton)] - ton).sum()
        check = pd.concat([off_times, on_times], axis=1, keys=["toff", "ton"])
        check = check.apply(counter, axis=1)
        counts.loc[check.index, "count"] = check
        return ConnectivityMatrix(self._edge_indices, vertex_properties=self._vertex_properties,
                                  edge_properties=counts, default_edge_property="count", shape=self._shape)
    
    def is_consistent(self):
        from scipy.spatial import distance
        def simple_diff(a, b):
            return (b - a)[0]

        def valid(arg):
            if not isinstance(arg["toff"], np.ndarray): return False
            if not isinstance(arg["ton"], np.ndarray): return len(arg["toff"]) == 1
            mat = distance.cdist(arg["ton"], arg["toff"], metric=simple_diff)
            return ~np.any(mat == 0) and\
                np.all(np.triu(mat, 1) >= 0) and\
                np.all(np.tril(mat) <= 0)
        
        off_times = self._off.reset_index().groupby("edge").apply(lambda x: np.vstack(x["t"]))
        on_times = self._on.reset_index().groupby("edge").apply(lambda x: np.vstack(x["t"]))
        check = pd.concat([off_times, on_times], axis=1, keys=["toff", "ton"])
        check = check.apply(valid, axis=1)
        return check

    def fix_consistency(self, copy=False):
        is_on = set(range(len(self._edge_indices)))
        valid_on = pd.Series(np.ones(len(self._on), dtype=bool), index=self._on.index)
        valid_off = pd.Series(np.ones(len(self._off), dtype=bool), index=self._off.index)

        N = np.maximum(self._off.index.max(), self._on.index.max())

        for i in range(N + 1):
            if i in self._off.index:
                valid_off[i] = np.isin(self._off[i], list(is_on))
                is_on.difference_update(self._off[i])
            if i in self._on.index:
                valid_on[i] = ~np.isin(self._on[i], list(is_on))
                is_on.update(self._on[i])
        
        if copy:
            return StructurallyPlasticMatrix(self._edge_indices, vertex_properties=self._vertex_properties,
                                             edge_properties=self._edges, default_edge_property=self._default_edge,
                                             shape=self._shape,
                                             edge_off=self._off[valid_off],
                                             edge_on=self._on[valid_on], check_consistency=False)
        
        self._on = self._on[valid_on]; self._off = self._off[valid_off]
        return self
    
    @classmethod
    def from_matrix_stack(cls, mats, vertex_labels=None, vertex_properties=None,
                          default_edge_property="data"):
        """
        Construct a StructurallyPlasticMatrix from a list of sparse matrices.
        """
        assert len(mats) > 0
        ms = [sparse.coo_matrix(_m) for _m in mats]
        assert np.all([_m.shape == ms[0].shape for _m in ms])
        shape = ms[0].shape
        
        def mat2df(mat):
            return pd.DataFrame({
                "row": mat.row, "col": mat.col
            })
        dfs = list(map(mat2df, ms))

        df = pd.concat(dfs, axis=0).drop_duplicates().reset_index(drop=True)
        curr_idx = pd.MultiIndex.from_frame(df)

        off_dict = {}; on_dict = {}; tent_off = []
        for t in range(len(dfs)):
            new_idx = pd.MultiIndex.from_frame(dfs[t])
            tent_on = tent_off
            tent_off = np.nonzero([_idx not in new_idx for _idx in curr_idx])[0]
            on_dict[t] = np.setdiff1d(tent_on, tent_off)
            off_dict[t] = np.setdiff1d(tent_off, tent_on)
        
        edge_properties = pd.DataFrame({default_edge_property: np.ones(len(df), dtype=bool)})
        return cls(df, vertex_labels=vertex_labels, vertex_properties=vertex_properties,
                   edge_properties=edge_properties, default_edge_property=default_edge_property,
                   shape=shape, edge_off=off_dict, edge_on=on_dict)


class TimeDependentMatrix(ConnectivityMatrix):
    """
    A version of ConnectivityMatrix for connectivity that changes over time functionally, i.e.
    unlike for `StructurallyPlasticMatrix` the presence and absence of edges remains fixed,
    but their efficacy (or weight) may change.
    """
    def __init__(self, *args, vertex_labels=None, vertex_properties=None,
                 edge_properties=None, default_edge_property=None, shape=None):
        """Not too intuitive init - please see `from_report()` below"""
        if len(args) == 1 and isinstance(args[0], np.ndarray) or isinstance(args[0], sparse.spmatrix):
            raise ValueError("TimeDependentMatrix can only be initialized by edge indices and edge properties")
        if isinstance(edge_properties, dict):
            assert np.all([x.columns.dtype == float for x in edge_properties.values()]),\
                 "Index of edge properties must be a float Index"
            edge_properties = pd.concat(edge_properties.values(), keys=edge_properties.keys(), names=["name"], axis=1)
            edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
        else:
            assert isinstance(edge_properties, pd.DataFrame)
            if isinstance(edge_properties.columns, pd.MultiIndex):
                assert len(edge_properties.columns.levels) == 2, "Columns must index time and name"
                if not edge_properties.columns.levels[0].dtype == float:
                    assert edge_properties.columns.levels[1].dtype == float,\
                        "Time index must be of type float Index"
                    edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
                else:
                    assert edge_properties.columns.levels[0].dtype == float,\
                        "Time index must be of type float Index"
            else:
                assert edge_properties.columns.dtype == float,\
                        "Time index must be of type Float64Index"
                edge_properties = pd.concat([edge_properties], axis=1, copy=False, keys=["agg_fn"], names=["name"])
                edge_properties.columns = edge_properties.columns.reorder_levels([1, 0])
        if default_edge_property is None:
            default_edge_property = edge_properties.columns.levels[1][0]
        self._times = np.unique(edge_properties.columns.get_level_values("time").to_numpy())
        self._time = self._times.min()
        super().__init__(*args, vertex_labels=vertex_labels, vertex_properties=vertex_properties,
                         edge_properties=edge_properties, default_edge_property=default_edge_property, shape=shape)

    @property
    def times(self):
        return self._times

    @property
    def edges(self):
        return self._edges[self._time]
    
    def at_time(self, new_time):
        # TODO: Add a copy=True kwarg that acts like in .default
        if new_time == -1:
            new_time = self._times.max()
        if new_time not in self._times:
            raise ValueError("No time point at {0} given".format(new_time))  # TODO: interpolate to nearest point?
        self._time = new_time
        return self

    def delta(self, t_fr, t_to):
        """Adds new `edge_property` (called 'delta') as the changes occuring between two time steps."""
        delta = self._edges[t_to] - self._edges[t_fr]
        for agg_fn in delta.columns.to_numpy():
            self.add_edge_property(("delta", agg_fn), delta[agg_fn].to_numpy())
        return self
    
    def default(self, new_default_property, copy=True):
        ret = super().default(new_default_property, copy=copy)
        if copy: ret._time = self._time
        return ret
    
    @classmethod
    def from_report(cls, sim, report_cfg, load_cfg, presyn_mapping=None):
        """
        A sonata synapse (compartment) report based constructor
        :param sim: `bluepysnap.Simulation` object
        :param report_cfg: config dict with report's name, time steps to load,
                           static property name to look up for synapses that aren't reported,
                           and the names of the aggregation functions to use
        :param load_cfg: config dict for loading and filtering neurons from the circuit
        :param presyn_mapping: mapping used to convert report from Neurodamus' post_gid & local_syn_id to
                               pre_gid and post_gid which can then be grouped and aggregated to get weighted connectomes
                               can be: pd.DataFrame with pre & post_gids and Neurodamus' local_syn_idx or
                                       filename of a saved DataFrame like that (loaded with `pd.read_pickle()`) or
                                       None (default) in which case the mapping will be calculated on the fly
        :return: a TimeDependentMatrix object
        """
        from .io.synapse_report import sonata_report, load_report, get_presyn_mapping, reindex_report, aggregate_data
        from .circuit_models.neuron_groups.grouping_config import load_filter

        nrn = load_filter(sim.circuit, load_cfg)
        lu_node_idx = pd.Series(range(len(nrn["node_ids"])), index=nrn["node_ids"])

        report, report_node_idx = sonata_report(sim, report_cfg)
        tgt_report_node_idx = np.intersect1d(nrn["node_ids"], report_node_idx)
        non_report_node_idx = np.setdiff1d(nrn["node_ids"], tgt_report_node_idx)
        data = load_report(report, report_cfg, tgt_report_node_idx)  # load only target (postsynaptic) node ids

        if presyn_mapping is None or len(tgt_report_node_idx) < len(report_node_idx):
            # the saved mapping is usually defined based on the full report, so if only parts are loaded
            # one would need to filter the mapping as well at which point, it's faster to just recalculate it
            presyn_mapping = get_presyn_mapping(sim.circuit, report_cfg["edge_population"], data.index)
        if not isinstance(presyn_mapping, pd.DataFrame):
            presyn_mapping = pd.read_pickle(presyn_mapping)

        data = reindex_report(data, presyn_mapping)
        data = data.iloc[data.index.get_level_values(0).isin(nrn["node_ids"])]  # filter to have only target pre_node_idx
        print("Report read! Starting aggregation of %i data points..." % data.shape[0])

        edges = aggregate_data(data, report_cfg, lu_node_idx)

        if len(non_report_node_idx) > 0:
            from .circuit_models import circuit_connection_matrix
            print("Looking up static values for %i non-reported postsynaptic neurons..." % len(non_report_node_idx))
            agg_funcs = list(edges.columns.levels[0])
            Ms = circuit_connection_matrix(sim.circuit, connectome=report_cfg["edge_population"],
                                           for_gids=nrn["node_ids"], for_gids_post=non_report_node_idx,
                                           edge_property=report_cfg["static_prop_name"], agg_func=agg_funcs)
            ts = edges.columns.levels[1]
            stat_edges = [pd.DataFrame.from_dict({t: Ms[agg_func].tocoo().data for t in ts})
                          for agg_func in agg_funcs]
            stat_edges = pd.concat(stat_edges, axis=1, copy=False,  keys=agg_funcs)
            stat_edges.columns.set_names(edges.columns.names, inplace=True)
            # map (non-reported, local) col idx to node ids and then back to (global) col idx
            lu_nr_node_idx = pd.Series(non_report_node_idx)
            stat_col_idx = lu_node_idx[lu_nr_node_idx[Ms[agg_funcs[0]].tocoo().col]].to_numpy()
            stat_edges.index = pd.MultiIndex.from_arrays(np.array([Ms[agg_funcs[0]].tocoo().row, stat_col_idx]),
                                                         names=["row", "col"])
            edges = pd.concat([edges, stat_edges])
            edges.sort_index(inplace=True)

        # separate the index from the edges (they're needed separately for the class' `init`)
        new_idx = pd.RangeIndex(len(edges))
        edge_idx = edges.index.to_frame().set_index(new_idx)
        edges.set_index(new_idx, inplace=True)
        edges.index.name = "edge_id"  # stupid pandas
        return cls(edge_idx, edge_properties=edges, vertex_properties=nrn.set_index("node_ids"),
                   shape=(len(nrn), len(nrn)))


class ConnectivityInSubgroups(ConnectivityMatrix):
    # TODO: This functionality can be in the main ConnectivityMatrix

    def __extract_vertex_ids__(self, an_obj):
        if isinstance(an_obj, str):
            assert self._vertex_properties[an_obj].dtype == bool, "Population spec must be a column of type bool"
            return self.gids[self._vertex_properties[an_obj]]

        if hasattr(an_obj, GID):
            return getattr(an_obj, GID)
        return an_obj


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
    def from_bluepy(cls, bluepy_obj, load_config=None, connectome=LOCAL_CONNECTOME, **kwargs):
        """
        Sonata config based constructor
        :param bluepy_obj: bluepysnap Simulation or Circuit object
        :param load_config: config dict for loading and filtering neurons from the circuit
        :param connectome: str. that can be "local" which specifies local circuit connectome
                           or the name of a projection to use
        Additional **kwargs are forwarded to a call of conntility.circuit_models.circuit_group_matrices
        """
        from .circuit_models.neuron_groups import load_group_filter
        from .circuit_models import circuit_group_matrices

        if hasattr(bluepy_obj, "circuit"):
            circ = bluepy_obj.circuit
        else:
            circ = bluepy_obj
        
        nrn = load_group_filter(circ, load_config)

        # TODO: think a bit about if it should even be possible to call this for a projection (remove arg. if not...)
        mats = circuit_group_matrices(circ, nrn, connectome=connectome, **kwargs)
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

