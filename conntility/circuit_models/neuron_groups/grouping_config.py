# SPDX-License-Identifier: Apache-2.0
import numpy
import pandas
import os

from . import make_groups
from . import load_neurons
from .from_atlas import atlas_property

def _read_if_needed(cfg_or_dict, resolve_at=None):
    if isinstance(cfg_or_dict, str) or isinstance(cfg_or_dict, os.PathLike):
        if not os.path.isfile(cfg_or_dict) and resolve_at is not None:
            cfg_or_dict = os.path.join(resolve_at, cfg_or_dict)
        import json
        resolve_at = os.path.split(os.path.abspath(cfg_or_dict))[0]
        with open(cfg_or_dict, "r") as fid:
            cfg = json.load(fid)
    else:
        cfg = cfg_or_dict
    cfg = _resolve_includes(cfg, resolve_at=resolve_at)
    return cfg

def _resolve_includes(cfg, resolve_at=None):
    if isinstance(cfg, dict):
        if "include" in cfg:
            return _read_if_needed(cfg["include"], resolve_at=resolve_at)
        for k, v in cfg.items():
            cfg[k] = _resolve_includes(v, resolve_at=resolve_at)
    elif isinstance(cfg, list):
       cfg = [_resolve_includes(v, resolve_at=resolve_at) for v in cfg]
    return cfg


def _flatten_nested_list(a_lst):
    assert isinstance(a_lst, list)
    for _ in range(len(a_lst)):
        e = a_lst.pop(0)
        if isinstance(e, list):
            _flatten_nested_list(e)
            a_lst.extend(e)
        else: a_lst.append(e)
    

def group_with_config(df_in, cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)

    if "grouping" in cfg:
        cfg = cfg["grouping"]
    if not isinstance(cfg, list):
        cfg = [cfg]

    is_first = True
    for grouping in cfg:
        if "name" in grouping and "filtering" in grouping:
            membership = evaluate_filter_config(df_in, grouping)
            idx_df = pandas.DataFrame(membership, columns=[grouping["name"]], index=df_in.index)
            if not is_first:
                idx_df = pandas.concat([df_in.index.to_frame(), idx_df], axis=1, copy=False)
            df_in = df_in.set_index(pandas.MultiIndex.from_frame(idx_df)).sort_index()
            is_first = False

        elif "method" in grouping:
            func = make_groups.__dict__.get(grouping["method"])
            if func is None:
                raise ValueError("Unknown grouping method: {0}".format(grouping["method"]))
            df_in = func(df_in, grouping["columns"], *grouping.get("args", []), replace=is_first,
                        **grouping.get("kwargs", {}))
            is_first = False
    return df_in

def evaluate_filter_config(df_in, cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)

    if "filtering" in cfg:
        cfg = cfg["filtering"]
    if not isinstance(cfg, list):
        cfg = [cfg]
    
    valid = numpy.ones(len(df_in), dtype=bool)
    for rule in cfg:
        if not "column" in rule:
            continue
        col = df_in[rule["column"]]
        if "values" in rule:
            valid = valid & numpy.in1d(col, rule["values"])
        elif "value" in rule:
            valid = valid & (col == rule["value"]).values
        elif "interval" in rule:
            iv = rule["interval"]
            assert len(iv) == 2, iv
            valid = valid & (col >= iv[0]).values & (col < iv[1]).values
    return valid


def filter_with_config(df_in, cfg_or_dict):
    valid = evaluate_filter_config(df_in, cfg_or_dict)
    return df_in.iloc[valid]


def filter_config_to_dict(cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)
    if "filtering" in cfg:
        cfg = cfg["filtering"]
    if not isinstance(cfg, list):
        cfg = [cfg]
    lst_tf = []
    for c in cfg:
        c = c.copy()
        lst_tf.append((c.pop("column"), str(list(c.values())[0])))  # Needs evaluation left to right
    return dict(lst_tf)


def load_with_config(circ, cfg_or_dict, node_population=None):
    cfg = _read_if_needed(cfg_or_dict)
    if "loading" in cfg:
        cfg = cfg["loading"]
    node_population = node_population or cfg.get("node_population", None)
    props = cfg.get("properties")
    if props is None:
        from .defaults import FLAT_COORDINATES, SS_COORDINATES
        if node_population is None:
            props = list(circ.nodes.property_names)
        else:
            props = list(circ.nodes[node_population].property_names)
    nrn = load_neurons(circ, props, cfg.get("base_target", None), node_population)

    atlases = cfg.get("atlas", [])
    try:
        _flatten_nested_list(atlases)
    except AssertionError:
        raise ValueError("Expected list here. Got: {0}".format(atlases.__class__))
    for atlas_spec in atlases:
        col_names = atlas_spec["properties"]
        if not isinstance(col_names, list): col_names = [col_names]
        nrn = pandas.concat([nrn, atlas_property(nrn, atlas_spec["data"], circ=circ, column_names=col_names)],
        axis=1, copy=False)
    
    groups = cfg.get("groups", [])
    try:
        _flatten_nested_list(groups)
    except AssertionError:
        raise ValueError("Expected list here. Got: {0}".format(groups.__class__))
    for group_spec in groups:
        nrn[group_spec["name"]] = evaluate_filter_config(nrn, group_spec)
        
    return nrn
    

def load_group_filter(circ, cfg_or_dict, node_population=None):
    cfg_or_dict = cfg_or_dict or {}
    ret = filter_with_config(
        group_with_config(
            load_with_config(circ, cfg_or_dict, node_population=node_population),
            cfg_or_dict
        ),
        cfg_or_dict
    )
    return ret


def load_filter(circ, cfg_or_dict, node_population=None):
    cfg_or_dict = cfg_or_dict or {}
    ret = filter_with_config(
        load_with_config(circ, cfg_or_dict, node_population=node_population),
        cfg_or_dict
    )
    return ret
