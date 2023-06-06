# SPDX-License-Identifier: Apache-2.0
import numpy

from tqdm import tqdm
from scipy.sparse import issparse


def similarity_matrix(C):
    """Turns a L x N connectivity matrix into a L x L similarity matrix. I.e. evaluates how similar the
    individual _rows_ of the input matrix are.
    Additionally returns a normalized version of the similarity that is relevant for the
    diffusion embedding process."""
    chunk_size = 100000
    C = C.astype(float)

    if issparse(C):
        C = C.tocsr()
        normalize_vals = numpy.array(numpy.sqrt(C.multiply(C).sum(axis=1))).flatten()
        for i, v in enumerate(normalize_vals): # Normalize connectivity
            C.data[C.indptr[i]:C.indptr[i+1]] = C.data[C.indptr[i]:C.indptr[i+1]] / v
        C = (C * C.transpose()).tocoo() # Similarity
        # TODO: After this, C might no longer really have a sparse structure. Might be better to 
        # cast .todense() and continue the non-sparse case from here.
        nrm = numpy.array(C.sum(axis=1)).flatten()
        chunks = numpy.arange(0, C.nnz + chunk_size, chunk_size)
        for ch_fr, ch_to in tqdm(zip(chunks[:-1], chunks[1:]), desc="Normalizing...", total=len(chunks) - 1):
            ratios = nrm[C.row[ch_fr:ch_to]] / nrm[C.col[ch_fr:ch_to]]
            C.data[ch_fr:ch_to] = (C.data[ch_fr:ch_to] / nrm[C.row[ch_fr:ch_to]]) / ratios

        if C.nnz > (0.2 * C.shape[0] * C.shape[1]):
            C = numpy.array(C.todense())
        else:
            C = C.tocsc()
    else:
        normalize_vals = numpy.linalg.norm(C, axis=1, keepdims=True)
        normalize_vals[normalize_vals == 0] = 1E-9 # In case it's zero
        C = C / normalize_vals # Normalize connectivity
        C = numpy.dot(C, C.transpose()) # Similarity
        cs_sum = C.sum(axis=1, keepdims=True)  # L x 1, zero for zero connected voxels
        cs_sum[cs_sum == 0] = 1E-9  # In case it's zero
        cs_sum_ratios = numpy.hstack([cs_sum / _x for _x in cs_sum[:, 0]])
        C = (C / cs_sum) * numpy.sqrt(cs_sum_ratios) # Normalize similarity
    return C


def embed_pathway(C, diffusion_time=1, n_components=3):
    """
    """
    try:
        from mapalign.embed import compute_diffusion_map
    except ImportError:
        raise RuntimeError("""This optional feature requires installation of the mapalign package.
        Obtain it from https://github.com/satra/mapalign""")
    print("Calculating similarity...")
    S_norm = similarity_matrix(C)
    print("...done! Performing diffusion mapping...")
    #  Treat voxels with zero connectivity, which would be considered disconnected from the rest of the graph
    vxl_is_valid = numpy.array(S_norm.sum(axis=1) != 0).flatten()
    embed_coords = numpy.NaN * numpy.ones((S_norm.shape[0], n_components), dtype=float)
    embed_coords[vxl_is_valid, :], embed_res = compute_diffusion_map(S_norm[numpy.ix_(vxl_is_valid, vxl_is_valid)],
                                                                     return_result=True,
                                                                     diffusion_time=diffusion_time,
                                                                     n_components=n_components)
    print("...done!")
    return S_norm, embed_coords, embed_res

