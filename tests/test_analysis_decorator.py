# SPDX-License-Identifier: Apache-2.0
from numpy.random.mtrand import RandomState
import pandas
import json
import os

from numpy import random
from scipy import sparse


from conntility import analysis as test_module
from conntility import ConnectivityMatrix


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

RANDOM_WORDS = ["banana", "apple", "orange", "tangerine", "car"]

sz = 500
nrn = pandas.DataFrame(
    random.randint(0, 5, size=(sz, 3)),
    columns=["mtype", "etype", "ttype"]
)
nrn["mtype"] = [RANDOM_WORDS[i] for i in nrn["mtype"]]
M = sparse.csc_matrix(random.rand(sz, sz) < 0.06)

with open(os.path.join(TEST_DATA_DIR, "test_analysis_config.json"), "r") as fid:
    ana_cfg_raw = json.load(fid)
with open(os.path.join(TEST_DATA_DIR, "test_analysis_control_sample.json"), "r") as fid:
    ana_cfg_sample = json.load(fid)

def test_analysis_by_config():
    ana_cfg = ana_cfg_raw.copy()
    rnd_er = ana_cfg["analyses"]["simplex_counts"]["source"] =\
        os.path.join(TEST_DATA_DIR, ana_cfg["analyses"]["simplex_counts"]["source"])
    rnd_er = ana_cfg["analyses"]["simplex_counts"]["decorators"][1]["analysis_arg"]["random_er"]
    rnd_er["source"] = os.path.join(TEST_DATA_DIR, rnd_er["source"])

    ana_cfg["analyses"]["random_number"]["source"] =\
        os.path.join(TEST_DATA_DIR, ana_cfg["analyses"]["random_number"]["source"])

    analyses = list(test_module.get_analyses(ana_cfg))

    for analysis in analyses:
        if analysis.name == "random_number":
            result = analysis.apply(M, nrn)
            assert len(result.index.names) == 2
            assert tuple(result.index.names) == ("idx-mtype", "idx-etype")
        elif analysis.name == "simplex_counts":
            result = analysis.apply(M, nrn).unstack("Control")
            nrm_result = (result["data"] - result["random_er"]) / result["random_er"]

            assert (nrm_result[:, 0] == 0).all()
            for word in RANDOM_WORDS:
                assert word in nrm_result
            assert nrm_result["orange"].index.name == "dim"


def test_manual_filtering_config():
    decorator = test_module.analysis_decorators.grouped_by_filtering_config
    fn_lst_filters = os.path.join(TEST_DATA_DIR, "test_list_filtering_config.json")
    with open(fn_lst_filters, "r") as fid:
        lst_filters = json.load(fid)

    base_function = lambda a, b: a.shape[0]
    result = decorator(lst_filters)(base_function)(M, nrn)
    assert len(result) == len(lst_filters)
    assert (nrn["etype"] == 2).sum() + len(nrn) == result.sum()  # Overlap in etype==2

    for fltr in lst_filters: fltr.pop("name")
    result = decorator(lst_filters)(base_function)(M, nrn)
    assert len(result) == len(lst_filters)
    assert (nrn["etype"] == 2).sum() + len(nrn) == result.sum()  # Overlap in etype==2


def test_analysis_with_control_sample():
    C = ConnectivityMatrix(M, vertex_properties=nrn)
    res = C.analyze(os.path.join(TEST_DATA_DIR, "test_analysis_control_sample.json"))
    sc = res["simplex_counts"]
    assert "sampled_by_mtype" in sc
    assert ("sampled_by_mtype", "in_assembly2") in sc
    assert ("sampled_by_mtype", "in_assembly2", 0) in sc
