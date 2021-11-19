from os import path
import json


def simulation_conditions(sim):
    sim_path = sim.config._path
    sim_json = sim_path + ".json"
    if not path.isfile(sim_json):
        return {}
    with open(sim_json, "r") as fid:
        conds = json.load(fid)
    return conds
