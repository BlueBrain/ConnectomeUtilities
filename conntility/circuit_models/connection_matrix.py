# For loading connectivity matrices
import h5py
import numpy
import tqdm
import pandas
from scipy import sparse

from .neuron_groups.defaults import GID

LOCAL_CONNECTOME = "local"
STR_VOID = "VOID"


def find_sonata_connectome(circ, connectome, return_sonata_file=True):
    """
    Returns the sonata connectome associated with a named projection; or the default 
    "local" connecome.
    Input:
    circ (bluepy.Circuit)
    connectome (str): Name of the projection to look up. Use "local" for the default local
    (i.e. touch-based) connectome.
    return_sonata_file (optional, default: True): If true, returns the path of the .sonata
    file. Else returns a bluepy.Connectome.
    """
    if return_sonata_file:
        if connectome == LOCAL_CONNECTOME:
            return circ.config["connectome"]
        return circ.config["projections"][connectome]
    if connectome == LOCAL_CONNECTOME:
        return circ.connectome
    return circ.projection[connectome]


def full_connection_matrix(sonata_fn, n_neurons=None, chunk=50000000):
    """
    Returns the full connection matrix encoded in a sonata h5 file.
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    n_neurons (optional): Number of neurons in the connectome. If not provided,
    it will be estimated from the connectivity info, but unconnected neurons may
    get ignored!
    chunk (optional): Number of connections to read at the same time. Larger values
    will run generally faster, but with fewer updates of the progress bar.

    Returns:
    scipy.sparse matrix of connectivity
    """
    h5 = h5py.File(sonata_fn, "r")['edges/default']
    if n_neurons is not None:
        n_neurons = (n_neurons, n_neurons)

    dset_sz = h5['source_node_id'].shape[0]
    A = numpy.zeros(dset_sz, dtype=int)
    B = numpy.zeros(dset_sz, dtype=int)
    splits = numpy.arange(0, dset_sz + chunk, chunk)
    for splt_fr, splt_to in tqdm.tqdm(zip(splits[:-1], splits[1:]), total=len(splits) - 1):
        A[splt_fr:splt_to] = h5['source_node_id'][splt_fr:splt_to]
        B[splt_fr:splt_to] = h5['target_node_id'][splt_fr:splt_to]
    M = sparse.coo_matrix((numpy.ones_like(A, dtype=bool), (A, B)), shape=n_neurons)
    return M.tocsr()


def connection_matrix_for_gids(sonata_fn, gids, gids_post=None):
    """
    Returns the connection matrix encoded in a sonata h5 file for a subset of neurons.
    Input:
    sonata_fn (str): Path to the sonata h5 file.
    gids: List of neuron gids to get the connectivity for.
    gids_post (optional): If given, then connectivity FROM gids TO gids_post will be
    returned. Else: FROM gids TO gids.

    Returns:
    scipy.sparse matrix of connectivity
    """
    idx = numpy.array(gids) - 1  # From gids to sonata "node" indices (base 0 instead of base 1)
    h5 = h5py.File(sonata_fn, "r")['edges/default']  # TODO: Instead of hard coding "default" that could be a config parameter
    if gids_post is None:
        gids_post = gids
    idx_post = numpy.array(gids_post) - 1
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
            row_ids = numpy.nonzero(numpy.in1d(idx, numpy.hstack(ids_pre)))[0]
            indices.extend(row_ids)
        indptr.append(len(indices))
    mat = sparse.csc_matrix((numpy.ones(len(indices), dtype=bool), indices, indptr), shape=(N, M))
    return mat


def circuit_connection_matrix(circ, connectome=LOCAL_CONNECTOME, for_gids=None, for_gids_post=None, chunk=50000000):
    """
    Returns a structural connection matrix, either for an entire circuit, or a subset of neurons.
    For either local connectivity or any projection. 
    Input:
    circ (bluepy.Circuit)
    connectome (str, default: "local"): Which connectome to return. Can be either "local", returning the 
    touch connectome or the name of any projection defined in the CircuitConfig.
    for_gids: List of neuron gids to get the connectivity for.
    for_gids_post (optional): If given, then connectivity FROM for_gids TO for_gids_post will be
    returned. Else: FROM for_gids TO for_gids. 
    NOTE: Can be used to get the matrix of external innervation!
    For that purpose, provide the gids of innervating fibers as for_gids, the gids of circuit neurons as
    for_gids_post and the name or the projection as connectome.
    chunk (optional): Number of connections to read at the same time. Larger values
    will run generally faster, but with fewer updates of the progress bar.

    Returns:
    scipy.sparse matrix of connectivity
    """
    conn_file = find_sonata_connectome(circ, connectome)
    N = circ.cells.count()
    if for_gids is None:
        return full_connection_matrix(conn_file, n_neurons=N, chunk=chunk)
    return connection_matrix_for_gids(conn_file, for_gids, gids_post=for_gids_post)


def circuit_group_matrices(circ, neuron_groups, connectome=LOCAL_CONNECTOME, extract_full=False,
                           column_gid=GID, **kwargs):
    """
    Returns matrices of the structural connectivity within specified groups of neurons.
    For either local connectivity or any projection. 
    Input:
    circ (bluepy.Circuit)
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
        neuron_groups = neuron_groups[GID]
    neuron_groups = neuron_groups.groupby(neuron_groups.index.names)
    if not extract_full:
        matrices = neuron_groups.apply(lambda grp: circuit_connection_matrix(circ, connectome=connectome,
                                                                             for_gids=grp.values, **kwargs))
    else:
        # TODO: Assumes the full matrix is index from gid 1 to N, which it should. But what if some gids are missing?
        full_matrix = circuit_connection_matrix(circ, connectome=connectome, **kwargs)
        matrices = neuron_groups.apply(lambda grp: full_matrix[numpy.ix_(grp.values - 1, grp.values - 1)])
    matrices = matrices[matrices.apply(lambda x: isinstance(x, sparse.spmatrix))]
    return matrices


def circuit_cross_group_matrices(circ, neuron_groups_pre, neuron_groups_post, connectome=LOCAL_CONNECTOME,
                                 extract_full=False, column_gid=GID, **kwargs):
    """
    Returns the structural connectivity between (and within) specified groups of neurons.
    That is, a number of matrices of structural connectivity between neurons in group A and B. 
    This can be thought of as one big matrix, broken up into sub-matrices by group.
    For either local connectivity or any projection. 
    Input:
    circ (bluepy.Circuit)
    neuron_groups_pre (pandas.DataFrame): Frame of neuron grouping info. 
    See conntility.circuit_models.neuron_groups for information how a group is defined. 
    These groups will be the groups _sending_ the connections that are considered.
    neuron_groups_post (pandas.DataFrame): Frame of neuron grouping info. 
    These groups will be the groups _receiving_ the connections that are considered. Can be the same as 
    neuron_groups_pre.
    connectome (str, default: "local"): Which connectome to return. Can be either "local", returning the 
    touch connectome or the name of any projection defined in the CircuitConfig.
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
                return full_matrix[numpy.ix_(df_pre[column_gid].values - 1, df_post[column_gid].values - 1)]
            return index_submat

        res = neuron_groups_pre.groupby(neuron_groups_pre.index.names).apply(
            lambda df_pre:
            neuron_groups_post.groupby(neuron_groups_post.index.names).apply(prepare_indexing(df_pre))
        )
        return res

    def prepare_con_mat(df_pre):
        _connectome = connectome
        if connectome is None:  # Fallback
            from .neuron_groups.defaults import PROJECTION
            assert PROJECTION in df_pre.index.names
            tmp = df_pre.index.to_frame()[PROJECTION].unique()
            assert len(tmp) == 1
            _connectome = tmp[0]

        def execute_con_mat(df_post):
            return circuit_connection_matrix(circ, for_gids=df_pre[column_gid].values,
                                             for_gids_post=df_post[column_gid].values,
                                             connectome=_connectome)

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
        all_gids = circ.cells.ids()
        missing_gids = numpy.setdiff1d(all_gids, node_lookup.index)
        full_lookup = pandas.concat([node_lookup,
                                     pandas.Series([STR_VOID] * len(missing_gids),
                                                   index=missing_gids)], axis=0)
        node_lookup = pandas.Series(pandas.Categorical(full_lookup), index=node_lookup.index, name=node_lookup.name)
    return node_lookup


def connection_matrix_between_groups_partition(sonata_fn, node_lookup, chunk=50000000):
    # TODO: If the user accidently provides a "neuron_groups" instead of "node_lookup" input give helpful message
    # TODO: Evaluate if it is necessary to fill node_lookup for unused gids with STR_VOID
    """
    Don't use this. Use circuit_matrix_between_groups
    """
    h5 = h5py.File(sonata_fn, "r")['edges/default']  # TODO: close file!

    dset_sz = h5['source_node_id'].shape[0]
    splits = numpy.arange(0, dset_sz + chunk, chunk)

    midxx = pandas.MultiIndex.from_tuples([], names=["Source node", "Target node"])
    counts = pandas.Series([], index=midxx, dtype=int)

    for splt_fr, splt_to in tqdm.tqdm(zip(splits[:-1], splits[1:]), desc="Counting...", total=len(splits) - 1):
        son_idx_fr = h5['source_node_id'][splt_fr:splt_to]
        son_idx_to = h5['target_node_id'][splt_fr:splt_to]
        reg_fr = node_lookup[son_idx_fr + 1]
        reg_to = node_lookup[son_idx_to + 1]
        new_counts = pandas.DataFrame({"Source node": reg_fr.values,
                                       "Target node": reg_to.values}).value_counts()
        counts = counts.add(new_counts, fill_value=0)

    for lvl, nm in zip(counts.index.levels, counts.index.names):
        if STR_VOID in lvl:
            counts = counts.drop(STR_VOID, level=nm)
    return counts


def _afferent_gids(h5, post_gid):
    rnge = h5["indices"]["target_to_source"]["node_id_to_ranges"][post_gid - 1]
    if rnge[1] == rnge[0]:
        return numpy.array([])
    son_idx_fr = [h5["source_node_id"][r[0]:r[1]]
                  for r in h5["indices"]["target_to_source"]["range_to_edge_id"][rnge[0]:rnge[1]]]
    son_idx_fr = numpy.hstack(son_idx_fr) + 1
    return son_idx_fr


def connection_matrix_between_groups_partial(sonata_fn, node_lookup, **kwargs):
    # TODO: If the user accidently provides a "neuron_groups" instead of "node_lookup" input give helpful message
    """
    Don't use this. Use circuit_matrix_between_groups
    """
    node_lookup = node_lookup[node_lookup != STR_VOID]
    gids_per_node = node_lookup.to_frame().groupby(node_lookup.name).apply(lambda x: x.index.values)

    lst_node_to = []
    lst_counts_from = []
    with h5py.File(sonata_fn, "r") as h5_file:
        h5 = h5_file['edges/default']
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


def circuit_matrix_between_groups(circ, neuron_groups, connectome=LOCAL_CONNECTOME,
                                  extract_full=False, column_gid=GID):
    """
    Returns the number of structural connections between (and within) specified groups of neurons.
    That is, single matrix of the _number_ of connections between group A and B.  
    This can be thought of as a connectome with reduced resolution, as it is similar to
    a voxelized connectome. In fact, if the neuron_groups are based on binned x, y, z coordinates,
    this essentially returns a voxelized connectome!
    For either local connectivity or any projection. 
    Input:
    circ (bluepy.Circuit)
    neuron_groups (pandas.DataFrame): Frame of neuron grouping info. 
    See conntility.circuit_models.neuron_groups for information how a group is defined.
    connectome (str, default: "local"): Which connectome to return. Can be either "local", returning the 
    touch connectome or the name of any projection defined in the CircuitConfig.
    extract_full (bool, default: False): If set to True, this will first extract the _complete_ connection
    matrix between all neurons, then look up and sum the relevant parts of that huge matrix. This can be faster,
    if the neuron_groups are a partition of all neurons, or close to it. But it uses much more memory.
    column_gid (str, default: "gid"): Name of the column of neuron_groups pre/post that holds the gid of a neuron.

    Returns:
    pandas.DataFrame of connection counts. Index of the frame will be a MultiIndex with the columns
    "Source node" and "Target node". Values of these entries will be strings generated from the 
    MultiIndex of neuron_groups. This is a bit awkward and will be improved in the future.
    Use .stack and .values to turn this output into a classic connection matrix (2d numpy array).

    Note: You could use this to count structural innervation strength from external innervation. But you would
    have to concatenate a group definition for the innervating fibers and a group definition for the circuit
    neurons. 
    """
    conn_file = find_sonata_connectome(circ, connectome)

    if extract_full:
        node_lookup = _make_node_lookup(circ, neuron_groups, column_gid)
        return connection_matrix_between_groups_partition(conn_file, node_lookup)
    else:
        node_lookup = _make_node_lookup(circ, neuron_groups, column_gid, fill_unused_gids=False)
        return connection_matrix_between_groups_partial(conn_file, node_lookup)
