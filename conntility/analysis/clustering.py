# SPDX-License-Identifier: Apache-2.0
import pandas
import numpy


def expand_partition(Mc, label_use="new_partition", label_original="_idxx_in_original"):
    L = Mc.vertices[label_original].apply(len).sum()
    lbls = -numpy.ones(L, dtype=int)
    for lbl, idxx in Mc.vertices.groupby(label_use)["_idxx_in_original"].apply(numpy.hstack).items():
        lbls[idxx] = lbl
    return lbls


def ren_eel(M, clust_funcs, clust_func_update, kmax=None, modularity_kwargs={}):
    """
    https://www.nature.com/articles/s41598-019-50739-3
    """
    if isinstance(clust_funcs, list):
        clust_funcs = dict([("base_partition_{0}".format(i), _func)
                            for i, _func in enumerate(clust_funcs)])
    if kmax is None: kmax = len(clust_funcs)
    assert kmax >= len(clust_funcs)

    func_names = list(clust_funcs.keys())
    for k, _func in clust_funcs.items():
        M.add_vertex_property(k, _func(M.matrix))
    
    scoreboard = pandas.Series([
        M.modularity(k, **modularity_kwargs) for k in func_names
    ], index=func_names)
    cand_count = 0

    while(True):
        Mc = M.condense(func_names)
        Mc.add_vertex_property("new_partition", clust_func_update(Mc.matrix))
        cand_str = "candidate_{0}".format(cand_count)
        cand_count += 1
        M.add_vertex_property(cand_str, expand_partition(Mc))

        if numpy.any([(M.vertices[_k] == M.vertices[cand_str]).all()
                    for _k in list(scoreboard.index)]):
            scoreboard.pop(scoreboard.idxmin())
        else:                
            cand_score = M.modularity(cand_str, **modularity_kwargs)
            if cand_score > scoreboard.min():
                if len(scoreboard) >= kmax:
                    scoreboard.pop(scoreboard.idxmin())
                scoreboard[cand_str] = cand_score
            else:
                scoreboard.pop(scoreboard.idxmin())
            
        print("""
        Iteration {0}:
            k = {1}
            min_score = {2}
            mean_score = {3}
            max_score = {4}

        """.format(
            cand_count - 1,
            scoreboard.min(),
            scoreboard.mean(),
            scoreboard.max()
        ))
        if len(scoreboard) == 1:
            break
    return scoreboard

    
                        

