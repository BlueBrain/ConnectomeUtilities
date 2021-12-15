from json import load
import pandas
import numpy
import os

from libsonata import ElementReportReader

AGG_FUNCS = {"sum": numpy.sum, "mean": numpy.mean}

def presyn_gid_lookup(presyn_mapping, target_gids):
    if not isinstance(presyn_mapping, pandas.DataFrame):
        presyn_mapping = pandas.read_pickle(presyn_mapping)

    lookup = presyn_mapping.set_index(["post_gid", "local_syn_idx"])["pre_gid"]
    sub_lookup = lookup.iloc[numpy.in1d(lookup, target_gids)]
    return sub_lookup


def report_fn(sim, report_cfg):
    h5f_name = os.path.join(sim.config["Run_Default"]["OutputRoot"],
                            "%s.h5" % report_cfg["report_name"])
    return h5f_name


def sonata_report(h5f_name):
    report = ElementReportReader(h5f_name)
    report = report[list(report.get_population_names())[0]]
    report_gids = numpy.asarray(report.get_node_ids()) + 1
    return report, report_gids


def _il_get(report, report_cfg, gids):
    """Fast, pure libsonata, in line implementation of report.get(`gids`)"""
    print("Reading report for {0} gids between {1} and {2}".format(len(gids), numpy.min(gids), numpy.max(gids)))
    view = report.get(node_ids=(gids-1).tolist(), tstart=report_cfg["t_start"], tstop=report_cfg["t_end"],
                      tstride=round(report_cfg["t_step"]/report.times[-1]))
    print("Read data of shape {0}".format(view.data.shape))
    col_idx = pandas.MultiIndex.from_tuples(view.ids, names=["post_gid", "local_syn_id"])
    col_idx = col_idx.set_levels(col_idx.levels[0]+1, level=0)  # get back gids from node_ids
    return pandas.DataFrame(data=view.data, index=pandas.Index(view.times, name="time"),
                        columns=col_idx).transpose()


def read_chunk(report, report_cfg, lookup, chunk_gids):
    data = _il_get(report, report_cfg, chunk_gids)  # read chunk
    idx = lookup.index.intersection(data.index)  # intersect with presyn gids of interest
    pre_gid = lookup[idx]  # look up presyn gids

    new_idx = idx.to_frame()  # create a new multi-index: pre_gid, post_gid
    new_idx["pre_gid"] = pre_gid
    data = data.loc[idx].set_index(pandas.MultiIndex.from_frame(new_idx[["pre_gid", "post_gid"]]))
    return data


def group_chunk(chunk, report_cfg):
    agg_funcs = report_cfg.get("aggregation", list(AGG_FUNCS.keys()))
    grp = chunk.groupby(level=[0, 1], sort=False, group_keys=False)
    res = pandas.concat([
        grp.apply(AGG_FUNCS[fun]) for fun in agg_funcs
        ],
        copy=False, axis=1, keys=agg_funcs, names=["data"]
    )
    return res


def group_chunked_data(inputs):
    data, report_cfg, lookup, lo_gids = inputs
    idx = lookup.index.intersection(data.index)  # intersect with presyn gids of interest
    pre_gid = lookup[idx]  # look up presyn gids

    new_idx = idx.to_frame()  # create a new multi-index: pre_gid, post_gid
    new_idx["pre_gid"] = pre_gid
    data = data.loc[idx].set_index(pandas.MultiIndex.from_frame(new_idx[["pre_gid", "post_gid"]]))
    data = group_chunk(data, report_cfg)
    midx_frame = data.index.to_frame()
    for col in midx_frame.columns:  # Just a loop over 2 columns. No big deal.
        midx_frame[col] = lo_gids[midx_frame[col]].values
    data.index = pandas.MultiIndex.from_frame(midx_frame)
    return data


def read_and_group_chunk(inputs):
    """
    Use "group_chunked_data" instead!
    """
    report_fn, report_cfg, lookup, lo_gids, tgt_report_gids = inputs
    report, report_gids = sonata_report(report_fn)
    chunk = read_chunk(report, report_cfg, lookup, tgt_report_gids)
    df = group_chunk(chunk, report_cfg)
    midx_frame = df.index.to_frame()
    for col in midx_frame.columns:  # Just a loop over 2 columns. No big deal.
        midx_frame[col] = lo_gids[midx_frame[col]].values
    df.index = pandas.MultiIndex.from_frame(midx_frame)
    return df


def to_time_df(data, time_stamps):
    pass
