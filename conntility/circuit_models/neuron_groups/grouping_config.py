import numpy

from . import make_groups
from . import load_neurons

def _read_if_needed(cfg_or_dict):
    if isinstance(cfg_or_dict, str):
        import json
        with open(cfg_or_dict, "r") as fid:
            cfg = json.load(cfg_or_dict)
    else:
        cfg = cfg_or_dict
    return cfg

def group_with_config(df_in, cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)

    if "grouping" in cfg:
        cfg = cfg["grouping"]
    if not isinstance(cfg, list):
        cfg = [cfg]

    is_first = True
    for grouping in cfg:
        func = make_groups.__dict__.get(grouping["method"])
        if func is None:
            raise ValueError("Unknown grouping method: {0}".format(grouping["method"]))
        df_in = func(df_in, grouping["columns"], *grouping.get("args", []), replace=is_first,
                     **grouping.get("kwargs", {}))
        is_first = True
    return df_in

def filter_with_config(df_in, cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)

    if "filtering" in cfg:
        cfg = cfg["filtering"]
    if not isinstance(cfg, list):
        cfg = [cfg]
    
    valid = numpy.ones(len(df_in), dtype=bool)
    for rule in cfg:
        col = df_in[rule["column"]]
        if "values" in rule:
            valid = valid & numpy.in1d(col, rule["values"])
        elif "value" in rule:
            valid = valid & (col == rule["value"]).values
        elif "interval" in rule:
            iv = rule["interval"]
            assert len(iv) == 2, iv
            valid = valid & (col >= iv[0]).values & (col < iv[1]).values
    return df_in.iloc[valid]


def load_with_config(circ, cfg_or_dict):
    cfg = _read_if_needed(cfg_or_dict)
    if "loading" in cfg:
        cfg = cfg["loading"]
    props = cfg.get("properties")
    if props is None:
        from .defaults import FLAT_COORDINATES, SS_COORDINATES
        props = list(circ.cells.available_properties) + FLAT_COORDINATES + SS_COORDINATES
    nrn = load_neurons(circ, props, cfg.get("base_target", None))
    return nrn
    

def load_group_filter(circ, cfg_or_dict):
    ret = filter_with_config(
        group_with_config(
            load_with_config(circ, cfg_or_dict),
            cfg_or_dict
        ),
        cfg_or_dict
    )
    return ret
