# SPDX-License-Identifier: Apache-2.0
"""
Neurodamus reindexes global sonata synapse indices and reports local (starting from 0 for every node id) synapse IDs
These function deal with such reports: load reports, create mappings between global and local synapse IDs
and reindexes report to pre_node_id - post_node_id (or row - col for sparse matrices)
authors: András Ecker, Sirio Bolaños-Puchet, Michael Reimann
last modified: 07.2023
"""

import os
import numpy as np
import pandas as pd
from libsonata import Selection, EdgeStorage, ElementReportReader


def _report_fn(sim, report_cfg):
    return os.path.join(sim.config["output"]["output_dir"], "%s.h5" % report_cfg["report_name"])


def sonata_report(sim, report_cfg):
    """Init. sonata report reader"""
    h5f_name = _report_fn(sim, report_cfg)
    report = ElementReportReader(h5f_name)
    report = report[list(report.get_population_names())[0]]
    report_node_idx = np.asarray(report.get_node_ids())
    return report, report_node_idx


def load_report(report, report_cfg, node_idx):
    """Load synapse report using `libsonata`"""
    print("Reading report for {0} node ids (between {1} and {2})".format(len(node_idx),
                                                                         np.min(node_idx), np.max(node_idx)))
    view = report.get(node_ids=(node_idx).tolist(), tstart=report_cfg["t_start"], tstop=report_cfg["t_end"],
                      tstride=round(report_cfg["t_step"]/report.times[-1]))
    # print("Read data of shape {0}".format(view.data.shape))
    col_idx = view.ids
    col_idx = pd.MultiIndex.from_arrays(col_idx.transpose(), names=["post_gid", "local_syn_idx"])
    return pd.DataFrame(data=view.data, index=pd.Index(view.times, name="time"), columns=col_idx).transpose()


def _edge_fn(c, report_cfg):
    for edges in c.config["networks"]["edges"]:
        if report_cfg["edge_population"] in edges["populations"]:
            return edges["edges_file"]
    raise RuntimeError("Edge file name corresponding to %s not found!" % report_cfg["edge_population"])


def _get_afferrent_global_syn_idx(h5f_name, node_idx):
    """Creates lookup for global/sonata synapse IDs: dict with node ids as keys
    and synapse ID Selections (that can be flatten to get all idx) as values"""
    edges = EdgeStorage(h5f_name)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    return {node_id: edge_pop.afferent_edges(int(node_id)) for node_id in node_idx}


def _get_afferrent_node_idx(h5f_name, global_syn_idx):
    """Reads pre gids corresponding to syn_idx from sonata edge file"""
    edges = EdgeStorage(h5f_name)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    return edge_pop.source_nodes(Selection(global_syn_idx))


def _local2global_syn_idx(syn_id_dict, node_id, local_syn_idx):
    """Maps local [node_id][syn_id] synapse ID to global [sonata_syn_id] synapse ID"""
    flat_global_syn_idx = syn_id_dict[node_id].flatten()
    return flat_global_syn_idx[local_syn_idx]


def get_presyn_mapping(c, connectome, local_syn_idx_mi, pklf_name=None):
    """Creates pandas DataFrame with global [sonata_syn_id] synapse ID as index
    and local [post_node_id][syn_id] synapse IDs (and pre_node_id) as columns"""
    from ..circuit_models.sonata_helpers import find_sonata_connectome
    h5f_name = find_sonata_connectome(c, connectome)
    report_node_idx = local_syn_idx_mi.get_level_values(0).to_numpy()
    local_syn_idx = local_syn_idx_mi.get_level_values(1).to_numpy()
    sort_idx = np.argsort(report_node_idx)
    report_node_idx, local_syn_idx = report_node_idx[sort_idx], local_syn_idx[sort_idx]
    unique_node_idx, start_idx, counts = np.unique(report_node_idx, return_index=True, return_counts=True)

    global_syn_idx_dict = _get_afferrent_global_syn_idx(h5f_name, unique_node_idx)
    global_syn_idx = np.zeros_like(local_syn_idx, dtype=np.int64)
    for node_id, start_id, count in zip(unique_node_idx, start_idx, counts):
        end_id = start_id + count
        global_syn_idx[start_id:end_id] = _local2global_syn_idx(global_syn_idx_dict, node_id,
                                                                local_syn_idx[start_id:end_id])

    sort_idx = np.argsort(global_syn_idx)
    global_syn_idx = global_syn_idx[sort_idx]
    pre_node_idx = _get_afferrent_node_idx(h5f_name, global_syn_idx)
    presyn_mapping = pd.DataFrame({"pre_node_id": pre_node_idx, "post_node_id": report_node_idx[sort_idx],
                                   "local_syn_idx": local_syn_idx[sort_idx]}, index=global_syn_idx)
    presyn_mapping.index.name = "global_syn_idx"  # stupid pandas...
    print("Presynaptic mapping of length %i created" % len(presyn_mapping))
    if pklf_name is not None:
        presyn_mapping.to_pickle(pklf_name)
    return presyn_mapping


def reindex_report(data, presyn_mapping):
    """Re-indexes synapse report from (Neurodamus style) post_gid & local_syn_idx MultiIndex
    to a more usefull pre_node_id & post_node_id MultiIndex"""
    assert len(data) == len(presyn_mapping), "data and mapping has to have the same number of rows" \
                                             "for the reindexing to work (as it's implemented atm.)"
    data.sort_index(inplace=True)  # sort to have the same ordering as the mapping df
    data.index = pd.MultiIndex.from_frame(presyn_mapping[["pre_node_id", "post_node_id"]])
    return data


def _reindex_agg_res(agg_res, lu_node_idx):
    """Re-indexes aggregated results from pre_node_id & post_node_id MultiIndex to sparse matrix row & col MultiIndex"""
    midx_frame = agg_res.index.to_frame()
    for col in midx_frame.columns:
        midx_frame[col] = lu_node_idx[midx_frame[col]].values
    agg_res.index = pd.MultiIndex.from_frame(midx_frame.rename(columns={"pre_node_id": "row", "post_node_id": "col"}))
    return agg_res


def aggregate_data(data, report_cfg, lu_node_idx):
    """Groups (synapses in the same connection together) and aggregates data"""
    agg_res = data.groupby(level=[0, 1], sort=False, group_keys=False).agg(report_cfg["aggregation"])
    # clean-up the new column idx a bit
    agg_res = agg_res.swaplevel(axis=1)
    agg_res.columns.set_names("agg_fn", level=0, inplace=True)
    agg_res.sort_index(axis=1, inplace=True)
    # clean-up the row idx as well
    return _reindex_agg_res(agg_res, lu_node_idx)
