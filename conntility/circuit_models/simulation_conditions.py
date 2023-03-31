from os import path
import json


def simulation_conditions(sim):
    """
    Return simulation conditions of a bluepy.Simulation, looked up from
    the assiciated BlueConfig.json
    """
    # TODO: UPDATE FOR SONATA!
    sim_path = sim.config._path
    sim_json = sim_path + ".json"
    if not path.isfile(sim_json):
        return {}
    with open(sim_json, "r") as fid:
        conds = json.load(fid)
    return conds
