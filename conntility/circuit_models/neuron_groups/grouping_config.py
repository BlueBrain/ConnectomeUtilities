import numpy

from . import make_groups
from . import load_neurons

def _read_if_needed(cfg_or_dict):
    if isinstance(cfg_or_dict, str):
        import json
        with open(cfg_or_dict, "r") as fid:
            cfg = json.load(fid)
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
        if not "method" in grouping:
            continue
        func = make_groups.__dict__.get(grouping["method"])
        if func is None:
            raise ValueError("Unknown grouping method: {0}".format(grouping["method"]))
        df_in = func(df_in, grouping["columns"], *grouping.get("args", []), replace=is_first,
                     **grouping.get("kwargs", {}))
        is_first = False
    return df_in

def filter_with_config(df_in, cfg_or_dict):
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
    cfg_or_dict = cfg_or_dict or {}
    ret = filter_with_config(
        group_with_config(
            load_with_config(circ, cfg_or_dict),
            cfg_or_dict
        ),
        cfg_or_dict
    )
    return ret


def load_filter(circ, cfg_or_dict):
    cfg_or_dict = cfg_or_dict or {}
    ret = filter_with_config(
        load_with_config(circ, cfg_or_dict),
        cfg_or_dict
    )
    return ret
