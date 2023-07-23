# SPDX-License-Identifier: Apache-2.0
# For loading connectivity matrices
import h5py
import numpy
import tqdm
import pandas
import os
from scipy import sparse

from .neuron_groups.defaults import GID
from .sonata_helpers import LOCAL_CONNECTOME, find_sonata_connectome, get_connectome_shape

STR_VOID = "VOID"


def full_connection_matrix(sonata_fn, n_neurons=None, population="default",
                           edge_property=None, agg_func=None, shape=None, chunk=50000000):
    """
    Returns the full connection matrix encoded in a sonata h5 file.
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    n_neurons (optional): Number of neurons in the connectome. If not provided, it will be estimated
                          from the connectivity info, but unconnected neurons may get ignored!
    population (str): Sonata population to work with.
    edge_property (str, optional): Name of a synapse property to look up. Must exist in the connectome.
    If not provided, a boolean matrix is returned
    agg_func (list, optional): Name of aggregation function to apply to the property of synapses belonging to the same connection.
    Must be provided if edge_property is provided!
    chunk (optional): Number of connections to read at the same time. Larger values
    will run generally faster, but with fewer updates of the progress bar.

    Returns:
    scipy.sparse matrix of connectivity
    """
    if edge_property is not None:
        assert agg_func is not None, "Must also provide a list of functions to aggregate synapses belonging to the same connection!"
        return _full_connection_property(sonata_fn, edge_property, agg_func, n_neurons=n_neurons,
                                         population=population, shape=shape, chunk=chunk)
    h5 = h5py.File(sonata_fn, "r")['edges/%s' % population]
    if n_neurons is not None and shape is None:
        shape = (n_neurons, n_neurons)

    dset_sz = h5['source_node_id'].shape[0]
    A = numpy.zeros(dset_sz, dtype=int)
    B = numpy.zeros(dset_sz, dtype=int)
    splits = numpy.arange(0, dset_sz + chunk, chunk)
    for splt_fr, splt_to in tqdm.tqdm(zip(splits[:-1], splits[1:]), total=len(splits) - 1):
        A[splt_fr:splt_to] = h5['source_node_id'][splt_fr:splt_to]
        B[splt_fr:splt_to] = h5['target_node_id'][splt_fr:splt_to]
    M = sparse.coo_matrix((numpy.ones_like(A, dtype=bool), (A, B)), shape=shape)
    return M.tocsr()


def _full_connection_property(sonata_fn, edge_property, agg_func, n_neurons=None, population="default",
                              shape=None, chunk=50000000):
    """
    Returns the full connection matrix encoded in a sonata h5 file. Instead of just a binary matrix, it looks up
    and assigns structural properties to the connections
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    edge_property (str): Name of the property to look up. Must exist in the connectome
    agg_func (list): Name of aggregation function to apply to the property of synapses belonging to the same connection
    n_neurons (optional): Number of neurons in the connectome. If not provided, it will be estimated
                          from the connectivity info, but unconnected neurons may get ignored!
    population (str): Sonata population to work with.
    chunk (optional): Number of connections to read at the same time. Larger values
    will run generally faster, but with fewer updates of the progress bar.

    Returns:
    scipy.sparse matrix of connectivity
    """
    if not isinstance(agg_func, list):
        raise NotImplementedError("Currently only implemented for list-of-functions")
    h5 = h5py.File(sonata_fn, "r")['edges/%s' % population]
    if n_neurons is not None and shape is None:
        shape = (n_neurons, n_neurons)

    dset_sz = h5['source_node_id'].shape[0]
    row_indices = []
    col_indices = []
    out_data = dict([(afunc, []) for afunc in agg_func])

    splits = numpy.arange(0, dset_sz + chunk, chunk)
    for splt_fr, splt_to in tqdm.tqdm(zip(splits[:-1], splits[1:]), total=len(splits) - 1):
        A = h5['source_node_id'][splt_fr:splt_to]
        B = h5['target_node_id'][splt_fr:splt_to]
        data = h5['0'][edge_property][splt_fr:splt_to]
        S = pandas.Series(data, index=pandas.MultiIndex.from_arrays([A, B])).groupby(level=[0, 1]).agg(agg_func)
        Si = S.index.to_frame()
        row_indices.extend(Si[0].values); col_indices.extend(Si[1].values)
        for afunc in agg_func:
            out_data[afunc].extend(S[afunc].values)

    M = dict([
        (afunc,
         sparse.coo_matrix((out_data[afunc], (row_indices, col_indices)), shape=shape).tocsr())
         for afunc in agg_func
    ])
    return M


def _connection_property_for_gids(sonata_fn, gids, gids_post, population, edge_property, agg_func):
    """
    Returns the connection matrix encoded in a sonata h5 file for a subset of neurons.
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    gids: List of neuron gids to get the connectivity for.
    gids_post (optional): If given, then connectivity FROM gids TO gids_post will be returned. Else: FROM gids TO gids.
    population (str): Sonata population to work with.
    edge_property (str): Name of the synapse property to report as the value of a connection
    agg_func: if a callable: A function to aggregate multiple synapse properties into a single,
                             scalar connection property (has to work with pandas' `.apply()`)
              if a list: A list of the above described aggregation functions (has to work with pandas' `.agg()`)

    Returns:
    scipy.sparse matrix of connectivity (or a dict of those if a list is passed as `agg_func`)
    """
    h5 = h5py.File(sonata_fn, "r")['edges/%s' % population]

    idx = numpy.array(gids)
    if gids_post is None:
        gids_post = gids
    idx_post = numpy.array(gids_post)
    rv_index = pandas.Series(range(len(idx)), index=idx)
    N = len(gids)
    M = len(gids_post)

    indices = []
    indptr = [0]
    if not isinstance(agg_func, list):
        data = []
        for id_post in tqdm.tqdm(idx_post):
            ids_pre = []
            data_pre = []
            ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
            for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
                ids_pre.append(h5['source_node_id'][block[0]:block[1]])
                data_pre.append(h5['0'][edge_property][block[0]:block[1]])

            if len(ids_pre) > 0:
                ids_pre = numpy.hstack(ids_pre)
                row_ids = rv_index[rv_index.index.intersection(ids_pre)].values
                indices.extend(row_ids)
                data_pre = pandas.Series(numpy.hstack(data_pre), index=ids_pre)
                data_pre = data_pre[data_pre.index.intersection(idx)]
                data.extend(data_pre.groupby(level=0, group_keys=False).apply(agg_func).values)

            indptr.append(len(indices))
        mat = sparse.csc_matrix((data, indices, indptr), shape=(N, M))
        return mat
    else:
        data = {agg_f: [] for agg_f in agg_func}
        for id_post in tqdm.tqdm(idx_post):
            ids_pre = []
            data_pre = []
            ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
            for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
                ids_pre.append(h5['source_node_id'][block[0]:block[1]])
                data_pre.append(h5['0'][edge_property][block[0]:block[1]])

            if len(ids_pre) > 0:
                ids_pre = numpy.hstack(ids_pre)
                row_ids = rv_index[rv_index.index.intersection(ids_pre)].values
                indices.extend(row_ids)
                data_pre = pandas.Series(numpy.hstack(data_pre), index=ids_pre)
                data_pre = data_pre[data_pre.index.intersection(idx)]
                # here is the main difference from the one below
                # (this is needed to be in sync. for `.io.synapse_report/aggregate_data()`)
                res = data_pre.groupby(level=0, group_keys=False).agg(agg_func)
                for agg_f in agg_func:
                    data[agg_f].extend(res[agg_f].to_numpy())

            indptr.append(len(indices))
        mats = {agg_f: sparse.csc_matrix((data[agg_f], indices, indptr), shape=(N, M)) for agg_f in agg_func}
        return mats


def connection_matrix_for_gids(sonata_fn, gids, gids_post=None, population="default",
                               edge_property=None, agg_func=None, load_full=False, **kwargs):
    """
    Returns the connection matrix encoded in a sonata h5 file for a subset of neurons.
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    gids: List of neuron gids to get the connectivity for.
    gids_post (optional): If given, then connectivity FROM gids TO gids_post will be returned. Else: FROM gids TO gids.
    population (str): Sonata population to work with.
    edge_property (optional, str): Name of the synapse property to report as the value of a connection
                                   If not provided, "True" will be reported for all connections.
                                   If provided, _must_ also provide `agg_func`.
    agg_func: if a callable: A function to aggregate multiple synapse properties into a single,
                             scalar connection property (has to work with pandas' `.apply()`)
              if a list: A list of the above described aggregation functions (has to work with pandas' `.agg()`)

    Returns:
    scipy.sparse matrix of connectivity (or a dict of those if a list is passed as `agg_func`)
    """
    if gids_post is None:
        gids_post = gids
    if load_full:
        M = full_connection_matrix(sonata_fn, population=population, edge_property=edge_property,
                                   agg_func=agg_func, **kwargs)
        if isinstance(M, dict): return dict([(k, v.tocsr()[numpy.ix_(gids, gids_post)])
                                             for k, v in M.items()])
        return M.tocsr()[numpy.ix_(gids, gids_post)]
    if edge_property is not None:
        assert agg_func is not None, "When looking up connection properties, must provide an agg_func, such as 'mean'"
        return _connection_property_for_gids(sonata_fn, gids, gids_post, population, edge_property, agg_func)
    
    h5 = h5py.File(sonata_fn, "r")['edges/%s' % population]
    idx = numpy.array(gids)
    idx_post = numpy.array(gids_post)

    rv_index = pandas.Series(range(len(idx)), index=idx)
    N = len(gids)
    M = len(gids_post)

    indices = []
    indptr = [0]
    for id_post in tqdm.tqdm(idx_post):
        ids_pre = []
        ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
        for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
            ids_pre.append(h5['source_node_id'][block[0]:block[1]])
        if len(ids_pre) > 0:
            row_ids = rv_index[rv_index.index.intersection(numpy.hstack(ids_pre))].values
            indices.extend(row_ids)
        indptr.append(len(indices))
    mat = sparse.csc_matrix((numpy.ones(len(indices), dtype=bool), indices, indptr), shape=(N, M))
    return mat


def circuit_connection_matrix(circ, connectome=LOCAL_CONNECTOME, for_gids=None, for_gids_post=None,
                              edge_population=None, node_population=None,
                              edge_property=None, agg_func=None, chunk=50000000, load_full=False):
    """
    Returns a structural connection matrix, either for an entire circuit, or a subset of neurons.
    For either local connectivity or any projection. 
    Input:
    circ (bluepysnap.Circuit)
    connectome (str): Which connectome to return. Can be any EdgePopulation in circ.edges. If not provided,
                      it will be heuristically guessed using the value of node_population. The first EdgePopulation
                      that has the specified node_population as both source and target will be used. If 
                      node_population is also not provided, then the largest non-virtual population will be used.
    for_gids: List of neuron gids to get the connectivity for.
    for_gids_post (optional): If given, then connectivity FROM for_gids TO for_gids_post will be returned.
                              Else: FROM for_gids TO for_gids.
                              NOTE: for_gids_post will be ignored if for_gids is not provided!
                              NOTE: Can be used to get the matrix of external innervation!
                              For that purpose, provide the gids of innervating fibers as for_gids,
                              the gids of circuit neurons as for_gids_post and the name or the projection as connectome.
    edge_population (str, optional): Sonata edge population name. Usually not required to be specified.
    node_population (str, optional): Name of a node population. Only used to look up local connectomes when
                              connectome="local" is used.
    edge_property (optional, str): Name of the synapse property to report as the value of a connection
                                   If not provided, "True" will be reported for all connections.
                                   If provided, _must_ also provide `agg_func`.
    agg_func: if a callable: A function to aggregate multiple synapse properties into a single,
                             scalar connection property (has to work with pandas' `.apply()`)
              if a list: A list of the above described aggregation functions (has to work with pandas' `.agg()`)
    chunk (optional): Number of connections to read at the same time. Larger values will run generally faster,
                      but with fewer updates of the progress bar.

    Returns:
    scipy.sparse matrix of connectivity (or a dict of those if a list is passed as `agg_func`)
    Note: By default, returns binary connectivity, i.e. only the presence or absence of at least one connection between
    nodes. If you want to count connections, use the edge_property=... with any valid edge property and 
    agg_func=len
    """
    if connectome == LOCAL_CONNECTOME: 
        from .sonata_helpers import local_connectome_for, nonvirtual_node_population
        if node_population is None: node_population = nonvirtual_node_population(circ)
        connectome = local_connectome_for(circ, node_population)
    conn_file = find_sonata_connectome(circ, connectome)
    shape = get_connectome_shape(circ, connectome)
    if edge_population is None:
        edge_population = connectome
    if for_gids is None:
        return full_connection_matrix(conn_file, edge_property=edge_property, agg_func=agg_func, shape=shape,
                                      population=edge_population, chunk=chunk)
    return connection_matrix_for_gids(conn_file, for_gids, gids_post=for_gids_post, population=edge_population,
                                      edge_property=edge_property, agg_func=agg_func, load_full=load_full, chunk=chunk)


def circuit_node_set_matrix(circ, for_node_set, for_node_set_post=None):
    """
    Returns a structural connection matrix within or between defined node sets. That is, unlike 
    circuit_connection_matrix this function can aggregate over multiple edge_populations and
    node_populations.
    """
    from .sonata_helpers import resolve_node_set

    if not isinstance(for_node_set, pandas.DataFrame):
        node_set_pre = resolve_node_set(circ, for_node_set).reset_index().set_index("population")
    else: node_set_pre = for_node_set.reset_index().set_index("population")
    if for_node_set_post is None: node_set_post = node_set_pre
    elif not isinstance(for_node_set_post, pandas.DataFrame):
        node_set_post = resolve_node_set(circ, for_node_set_post).reset_index().set_index("population")
    else: node_set_post = for_node_set_post.reset_index().set_index("population")

    row = []; col = []; data = []
    for edge_name, edges in circ.edges.items():
        rel_pre = edges.source.name in node_set_pre.index
        rel_post = edges.target.name in node_set_post.index
        if rel_pre and rel_post:
            tM = circuit_connection_matrix(circ, connectome=edge_name,
                                           for_gids=node_set_pre["node_ids"][edges.source.name],
                                           for_gids_post=node_set_post["node_ids"][edges.target.name]).tocoo()
            tgt_ids = node_set_post["index"][edges.target.name]
            src_ids = node_set_pre["index"][edges.source.name]
            
            row.extend(src_ids.iloc[tM.row])
            col.extend(tgt_ids.iloc[tM.col])
            data.extend(tM.data)
    M_out = sparse.coo_matrix((data, (row, col)), shape=(len(node_set_pre), len(node_set_post)))
    node_set_pre = node_set_pre.reset_index().set_index("index")
    node_set_post = node_set_post.reset_index().set_index("index")
    return M_out, node_set_pre, node_set_post


def circuit_group_matrices(circ, neuron_groups, connectome=LOCAL_CONNECTOME, extract_full=False,
                           column_gid=GID, **kwargs):
    """
    Returns matrices of the structural connectivity within specified groups of neurons.
    For any EdgePopulation (specified as connectome=...).

    Note: This function strongly assumes that the source and target node population of the specified
    connectome match the nodes specified in neuron_groups. No check is performed!

    Input:
    circ (bluepysnap.Circuit)
    neuron_groups (pandas.DataFrame): Frame of neuron grouping info. 
    See conntility.circuit_models.neuron_groups for information how a group is defined.
    connectome (str, default: "local"): Which connectome to return. Can be either "local", returning the 
    touch connectome or the name of any projection defined in the CircuitConfig.
    extract_full (bool, default: False): If set to True, this will first extract the _complete_ connection
    matrix between all neurons, then look up the relevant parts of that huge matrix. This can be faster,
    if the neuron_groups are a partition of all neurons, or close to it. But it uses much more memory.
    column_gid (str, default: "gid"): Name of the column of neuron_groups that holds the gid of a neuron.

    Returns:
    pandas.DataFrame of scipy.sparse matrices of connectivity. Index of the frame is identical to the index
    of neuron_groups.
    """
    if isinstance(neuron_groups, pandas.DataFrame):
        neuron_groups = neuron_groups[column_gid]
    neuron_groups = neuron_groups.groupby(neuron_groups.index.names)
    if not extract_full:
        matrices = neuron_groups.apply(lambda grp: circuit_connection_matrix(circ, connectome=connectome,
                                                                             for_gids=grp.values, **kwargs))
    else:
        full_matrix = circuit_connection_matrix(circ, connectome=connectome, **kwargs)
        matrices = neuron_groups.apply(lambda grp: full_matrix[numpy.ix_(grp.values, grp.values)])
    matrices = matrices[matrices.apply(lambda x: isinstance(x, sparse.spmatrix))]
    return matrices


def circuit_cross_group_matrices(circ, neuron_groups_pre, neuron_groups_post, connectome=LOCAL_CONNECTOME,
                                 extract_full=False, column_gid=GID, **kwargs):
    """
    Returns the structural connectivity between (and within) specified groups of neurons.
    That is, a number of matrices of structural connectivity between neurons in group A and B. 
    This can be thought of as one big matrix, broken up into sub-matrices by group.
    For any EdgePopulation (specified as connectome=...).

    Note: This function strongly assumes that the source and target of the EdgePopulation match the 
    nodes given in neuron_groups_pre and neuron_groups_post. No check is performed!

    Input:
    circ (bluepy.Circuit)
    neuron_groups_pre (pandas.DataFrame): Frame of neuron grouping info. 
    See conntility.circuit_models.neuron_groups for information how a group is defined. 
    These groups will be the groups _sending_ the connections that are considered.
    neuron_groups_post (pandas.DataFrame): Frame of neuron grouping info. 
    These groups will be the groups _receiving_ the connections that are considered. Can be the same as 
    neuron_groups_pre.
    connectome (str, default: "local"): Which connectome to return. Can be any EdgePopulation in circ.edges.
    If "local", a local recurrent connectome will be heuristically guessed. In that case, it is best to
    also specify the name of a node population using node_population=...
    The way a "local" connectome is guessed from node_population=... is documented in circuit_connection_matrix()
    extract_full (bool, default: False): If set to True, this will first extract the _complete_ connection
    matrix between all neurons, then look up the relevant parts of that huge matrix. This can be faster,
    if the neuron_groups are a partition of all neurons, or close to it. But it uses much more memory.
    column_gid (str, default: "gid"): Name of the column of neuron_groups pre/post that holds the gid of a neuron.

    Returns:
    pandas.DataFrame of scipy.sparse matrices of connectivity. Index of the frame is identical to the index
    of neuron_groups_pre. Columns are indentical to the index of neuron_groups_post.
    """
    if extract_full:
        full_matrix = circuit_connection_matrix(circ, connectome=connectome, **kwargs)

        def prepare_indexing(df_pre):
            def index_submat(df_post):
                return full_matrix[numpy.ix_(df_pre[column_gid].values, df_post[column_gid].values)]
            return index_submat

        res = neuron_groups_pre.groupby(neuron_groups_pre.index.names).apply(
            lambda df_pre:
            neuron_groups_post.groupby(neuron_groups_post.index.names).apply(prepare_indexing(df_pre))
        )
        return res

    def prepare_con_mat(df_pre):
        def execute_con_mat(df_post):
            if len(df_pre) == 0 or len(df_post) == 0:
                _shape = (len(df_pre), len(df_post))
                return sparse.csc_matrix(numpy.empty(shape=_shape, dtype=bool))
            return circuit_connection_matrix(circ, for_gids=df_pre[column_gid].values,
                                             for_gids_post=df_post[column_gid].values,
                                             connectome=connectome, **kwargs)

        return execute_con_mat

    res = neuron_groups_pre.groupby(neuron_groups_pre.index.names).apply(
        lambda df_pre:
        neuron_groups_post.groupby(neuron_groups_post.index.names).apply(prepare_con_mat(df_pre))
    )
    return res


def _make_node_lookup(circ, neuron_groups, column_gid, fill_unused_gids=True):
    from .neuron_groups import flip
    node_lookup = flip(neuron_groups, index=column_gid, contract_values=True, categorical=~fill_unused_gids)
    if fill_unused_gids:
        node_ids = circ.nodes.ids()
        all_gids = node_ids.index.to_frame()[GID].values
        missing_gids = numpy.setdiff1d(all_gids, node_lookup.index)
        full_lookup = pandas.concat([node_lookup,
                                     pandas.Series([STR_VOID] * len(missing_gids),
                                                   index=missing_gids)], axis=0)
        node_lookup = pandas.Series(pandas.Categorical(full_lookup), index=full_lookup.index, name=node_lookup.name)
    return node_lookup


def connection_matrix_between_groups_partition(sonata_fn, node_lookup, population, chunk=50000000):
    # TODO: If the user accidently provides a "neuron_groups" instead of "node_lookup" input give helpful message
    # TODO: Evaluate if it is necessary to fill node_lookup for unused gids with STR_VOID
    """
    Don't use this. Use circuit_matrix_between_groups
    """
    h5 = h5py.File(sonata_fn, "r")['edges/{0}'.format(population)]  # TODO: close file!

    dset_sz = h5['source_node_id'].shape[0]
    splits = numpy.arange(0, dset_sz + chunk, chunk)

    midxx = pandas.MultiIndex.from_tuples([], names=["Source node", "Target node"])
    counts = pandas.Series([], index=midxx, dtype=int)

    for splt_fr, splt_to in tqdm.tqdm(zip(splits[:-1], splits[1:]), desc="Counting...", total=len(splits) - 1):
        son_idx_fr = h5['source_node_id'][splt_fr:splt_to]
        son_idx_to = h5['target_node_id'][splt_fr:splt_to]
        reg_fr = node_lookup[son_idx_fr]
        reg_to = node_lookup[son_idx_to]
        new_counts = pandas.DataFrame({"Source node": reg_fr.values,
                                       "Target node": reg_to.values}).value_counts()
        counts = counts.add(new_counts, fill_value=0)

    for lvl, nm in zip(counts.index.levels, counts.index.names):
        if STR_VOID in lvl:
            counts = counts.drop(STR_VOID, level=nm)
    return counts


def _afferent_gids(h5, post_gid):
    rnge = h5["indices"]["target_to_source"]["node_id_to_ranges"][post_gid]
    if rnge[1] == rnge[0]:
        return numpy.array([])
    son_idx_fr = [h5["source_node_id"][r[0]:r[1]]
                  for r in h5["indices"]["target_to_source"]["range_to_edge_id"][rnge[0]:rnge[1]]]
    son_idx_fr = numpy.hstack(son_idx_fr)
    return son_idx_fr


def connection_matrix_between_groups_partial(sonata_fn, node_lookup, population="default", **kwargs):
    # TODO: If the user accidently provides a "neuron_groups" instead of "node_lookup" input give helpful message
    """
    Don't use this. Use circuit_matrix_between_groups
    """
    node_lookup = node_lookup[node_lookup != STR_VOID]
    gids_per_node = node_lookup.to_frame().groupby(node_lookup.name).apply(lambda x: x.index.values)

    lst_node_to = []
    lst_counts_from = []
    with h5py.File(sonata_fn, "r") as h5_file:
        h5 = h5_file['edges/{0}'.format(population)]
        for node_to, lst_post_gids in tqdm.tqdm(gids_per_node.items(), total=len(gids_per_node)):
            lst_pre_gids = [_afferent_gids(h5, post_gid) for post_gid in lst_post_gids]
            lst_pre_gids = numpy.hstack(lst_pre_gids)
            lst_pre_gids = lst_pre_gids[numpy.in1d(lst_pre_gids, node_lookup.index)]
            node_from = node_lookup[lst_pre_gids]
            counts_from = node_from.value_counts()
            counts_from.index.name = "Source node"
            lst_node_to.append(node_to)
            lst_counts_from.append(counts_from)
    counts = pandas.concat(lst_counts_from, keys=lst_node_to, names=["Target node"])

    return counts


def circuit_matrix_between_groups(circ, neuron_groups, connectome,
                                  edge_population=None, extract_full=False, column_gid=GID):
    """
    Returns the number of structural connections between (and within) specified groups of neurons.
    That is, single matrix of the _number_ of connections between group A and B.  
    This can be thought of as a connectome with reduced resolution, as it is similar to
    a voxelized connectome. In fact, if the neuron_groups are based on binned x, y, z coordinates,
    this essentially returns a voxelized connectome!
    For any EdgePopulation in circ.edges.

    Note: This function strongly assumes that the source and target of the specified connectome match the
    neurons given in neuron_groups. It is up to the user to ensure that!
    Note: If there are multiple connection (synapses) between a pair of nodes, they will be counted multiple
    times!

    Input:
    circ (bluepysnap.Circuit)
    neuron_groups (pandas.DataFrame): Frame of neuron grouping info. 
    See conntility.circuit_models.neuron_groups for information how a group is defined.
    connectome (str): Which connectome to return. Must be in circ.edges.
    edge_population (str, optional): Name of the edge population in the connectome. Usually not needed to be specified.
    extract_full (bool, default: False): If set to True, this will first extract the _complete_ connection
    matrix between all neurons, then look up and sum the relevant parts of that huge matrix. This can be faster,
    if the neuron_groups are a partition of all neurons, or close to it. But it uses much more memory.
    column_gid (str, default: "node_ids"): Name of the column of neuron_groups pre/post that holds the gid of a neuron.

    Returns:
    pandas.DataFrame of connection counts. Index of the frame will be a MultiIndex with the columns
    "Source node" and "Target node". Values of these entries will be strings generated from the 
    MultiIndex of neuron_groups. This is a bit awkward and will be improved in the future.
    Use .stack and .values to turn this output into a classic connection matrix (2d numpy array).

    Note: You could use this to count structural innervation strength from external innervation. But you would
    have to concatenate a group definition for the innervating fibers and a group definition for the circuit
    neurons. 
    """
    # TODO: Support "local" connectome?
    # TODO: Support non-recurrent connectome!
    conn_file = find_sonata_connectome(circ, connectome, assert_is_recurrent=True)
    if edge_population is None:
        edge_population = connectome

    if extract_full:
        node_lookup = _make_node_lookup(circ, neuron_groups, column_gid)
        return connection_matrix_between_groups_partition(conn_file, node_lookup, population=edge_population)
    else:
        node_lookup = _make_node_lookup(circ, neuron_groups, column_gid, fill_unused_gids=False)
        return connection_matrix_between_groups_partial(conn_file, node_lookup, population=edge_population)
