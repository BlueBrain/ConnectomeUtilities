import h5py
import numpy
import tqdm
from scipy import sparse


LOCAL_CONNECTOME = "local"


def find_sonata_connectome(circ, connectome):
    if connectome == LOCAL_CONNECTOME:
        return circ.config["connectome"]
    return circ.config["projections"][connectome]


def full_connection_matrix(sonata_fn, n_neurons=None, chunk=50000000):
    h5 = h5py.File(sonata_fn, "r")['edges/default']
    if n_neurons is not None:
        n_neurons = (n_neurons, n_neurons)

    dset_sz = h5['source_node_id'].shape[0]
    A = numpy.zeros(dset_sz, dtype=int)
    B = numpy.zeros(dset_sz, dtype=int)
    splits = numpy.arange(0, dset_sz + chunk, chunk)
    for splt_fr, splt_to in tqdm(zip(splits[:-1], splits[1:]), total=len(splits) - 1):
        A[splt_fr:splt_to] = h5['source_node_id'][splt_fr:splt_to]
        B[splt_fr:splt_to] = h5['target_node_id'][splt_fr:splt_to]
    M = sparse.coo_matrix((numpy.ones_like(A, dtype=bool), (A, B)), shape=n_neurons)
    return M.tocsr()


def circuit_connection_matrix(circ, connectome=LOCAL_CONNECTOME, for_gids=None, chunk=50000000):
    conn_file = find_sonata_connectome(circ, connectome)
    N = circ.cells.count()
    if for_gids is None:
        return full_connection_matrix(conn_file, n_neurons=N, chunk=chunk)
    return connection_matrix_for_gids(conn_file, for_gids)


def connection_matrix_for_gids(sonata_fn, gids):
    # TODO: Separate gids_pre, gids_post
    idx = numpy.array(gids) - 1  # From gids to sonata "node" indices (base 0 instead of base 1)
    h5 = h5py.File(sonata_fn, "r")['edges/default']  # TODO: Instead of hard coding "default" that could be a config parameter
    N = len(gids)

    indices = []
    indptr = [0]
    for id_post in tqdm(idx):
        ids_pre = []
        ranges = h5['indices']['target_to_source']['node_id_to_ranges'][id_post, :]
        for block in h5['indices']['target_to_source']['range_to_edge_id'][ranges[0]:ranges[1], :]:
            ids_pre.append(h5['source_node_id'][block[0]:block[1]])
        if len(ids_pre) > 0:
            row_ids = numpy.nonzero(numpy.in1d(idx, numpy.hstack(ids_pre)))[0]
            indices.extend(row_ids)
        indptr.append(len(indices))
    mat = sparse.csc_matrix((numpy.ones(len(indices), dtype=bool), indices, indptr), shape=(N, N))
    return mat
