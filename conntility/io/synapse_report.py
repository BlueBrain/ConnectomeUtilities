"""
Neurodamus reindexes global sonata synapse indices and reports local (starting from 0 for every gid) synapse IDs
These function deal with such reports: load reports, create mappings between global and local synapse IDs
and reindexes report to pre_gid - post_gid (or row - col for sparse matrices)
authors: András Ecker, Sirio Bolaños-Puchet, Michael Reimann
last modified: 01.2022
"""

import os
import numpy as np
import pandas as pd
from libsonata import Selection, EdgeStorage, ElementReportReader

AGG_FUNCS = ["mean", "sum"]


def _report_fn(sim, report_cfg):
    return os.path.join(sim.config["Run_Default"]["OutputRoot"], "%s.h5" % report_cfg["report_name"])


def sonata_report(sim, report_cfg):
    """Init. sonata report reader"""
    h5f_name = _report_fn(sim, report_cfg)
    report = ElementReportReader(h5f_name)
    report = report[list(report.get_population_names())[0]]
    report_gids = np.asarray(report.get_node_ids()) + 1
    return report, report_gids


def load_report(report, report_cfg, gids):
    """Fast, pure libsonata, in line implementation of report.get(`gids`)"""
    print("Reading report for {0} gids (between {1} and {2})".format(len(gids), np.min(gids), np.max(gids)))
    view = report.get(node_ids=(gids-1).tolist(), tstart=report_cfg["t_start"], tstop=report_cfg["t_end"],
                      tstride=round(report_cfg["t_step"]/report.times[-1]))
    # print("Read data of shape {0}".format(view.data.shape))
    col_idx = view.ids
    col_idx[:, 0] += 1  # get back gids from node_ids
    col_idx = pd.MultiIndex.from_arrays(col_idx.transpose(), names=["post_gid", "local_syn_idx"])
    return pd.DataFrame(data=view.data, index=pd.Index(view.times, name="time"), columns=col_idx).transpose()


def _get_afferrent_global_syn_idx(sonata_fn, gids):
    """Creates lookup for global/sonata synapse IDs: dict with (1-based) gids as keys
    and synapse ID Selections (that can be flatten to get all idx) as values"""
    edges = EdgeStorage(sonata_fn)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # get (global) afferent synapse idx (from 0-based sonata nodes)
    return {gid: edge_pop.afferent_edges(int(gid - 1)) for gid in gids}


def _get_afferrent_gids(sonata_fn, global_syn_idx):
    """Reads pre gids corresponding to syn_idx from sonata edge file"""
    edges = EdgeStorage(sonata_fn)
    edge_pop = edges.open_population(list(edges.population_names)[0])
    # get afferent (0-based sonata) node idx (+1 := (1-based) pre gids)
    return edge_pop.source_nodes(Selection(global_syn_idx)) + 1


def _local2global_syn_idx(syn_id_map, gid, local_syn_idx):
    """Maps local [gid][syn_id] synapse ID to global [sonata_syn_id] synapse ID"""
    flat_global_syn_idx = syn_id_map[gid].flatten()
    return flat_global_syn_idx[local_syn_idx]


def get_presyn_mapping(sonata_fn, local_syn_idx_mi, pklf_name=None):
    """Creates pandas DataFrame with global [sonata_syn_id] synapse ID as index
    and local [post_gid][syn_id] synapse IDs (and pre_gid) as columns"""
    report_gids = local_syn_idx_mi.get_level_values(0).to_numpy()
    local_syn_idx = local_syn_idx_mi.get_level_values(1).to_numpy()
    sort_idx = np.argsort(report_gids)
    report_gids, local_syn_idx = report_gids[sort_idx], local_syn_idx[sort_idx]
    unique_gids, start_idx, counts = np.unique(report_gids, return_index=True, return_counts=True)

    global_syn_idx_dict = _get_afferrent_global_syn_idx(sonata_fn, unique_gids)
    global_syn_idx = np.zeros_like(local_syn_idx, dtype=np.int64)
    for gid, start_id, count in zip(unique_gids, start_idx, counts):
        end_id = start_id + count
        global_syn_idx[start_id:end_id] = _local2global_syn_idx(global_syn_idx_dict, gid, local_syn_idx[start_id:end_id])

    sort_idx = np.argsort(global_syn_idx)
    global_syn_idx = global_syn_idx[sort_idx]
    pre_gids = _get_afferrent_gids(sonata_fn, global_syn_idx)
    presyn_mapping = pd.DataFrame({"pre_gid": pre_gids, "post_gid": report_gids[sort_idx],
                                   "local_syn_idx": local_syn_idx[sort_idx]}, index=global_syn_idx)
    presyn_mapping.index.name = "global_syn_idx"  # stupid pandas...
    print("Presynaptic mapping of length %i created" % len(presyn_mapping))
    if pklf_name is not None:
        presyn_mapping.to_pickle(pklf_name)
    else:
        return presyn_mapping


def reindex_report(data, presyn_mapping):
    """Re-indexes synapse report from (Neurodamus style) post_gid & local_syn_idx MultiIndex
    to a more usefull pre_gid & post_gid MultiIndex"""
    assert len(data) == len(presyn_mapping), "data and mapping has to have the same number of rows" \
                                             "for the reindexing to work (as it's implemented atm.)"
    data.sort_index(inplace=True)  # sort to have the same ordering as the mapping df
    data.index = pd.MultiIndex.from_frame(presyn_mapping[["pre_gid", "post_gid"]])
    return data


def _reindex_agg_res(agg_res, lo_gids):
    """Re-indexes aggregated results from pre_gid & post_gid MultiIndex to sparse matrix row & col MultiIndex"""
    midx_frame = agg_res.index.to_frame()
    for col in midx_frame.columns:
        midx_frame[col] = lo_gids[midx_frame[col]].values
    agg_res.index = pd.MultiIndex.from_frame(midx_frame.rename(columns={"pre_gid": "row", "post_gid": "col"}))
    return agg_res


def aggregate_data(data, report_cfg, lo_gids):
    """Groups (synapses in the same connection together) and aggregates (by default mean and sum) data"""
    agg_funcs = report_cfg.get("aggregation", AGG_FUNCS)
    agg_res = data.groupby(level=[0, 1], sort=False, group_keys=False).agg(agg_funcs)
    # clean-up the new column idx a bit
    agg_res = agg_res.swaplevel(axis=1)
    agg_res.columns.set_names("data", level=0, inplace=True)
    agg_res.sort_index(axis=1, inplace=True)
    # clean-up the row idx as well
    return _reindex_agg_res(agg_res, lo_gids)
