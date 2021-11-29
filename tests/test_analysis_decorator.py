from numpy.random.mtrand import RandomState
import pandas
import json
import os

from numpy import random
from scipy import sparse


from conntility import analysis as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")

RANDOM_WORDS = ["banana", "apple", "orange", "tangerine", "car"]

def test_analysis_by_config():
    sz = 500
    nrn = pandas.DataFrame(
        random.randint(0, 5, size=(sz, 3)),
        columns=["mtype", "etype", "ttype"]
    )
    nrn["mtype"] = [RANDOM_WORDS[i] for i in nrn["mtype"]]
    M = sparse.csc_matrix(random.rand(sz, sz) < 0.06)

    with open(os.path.join(TEST_DATA_DIR, "test_analysis_config.json"), "r") as fid:
        ana_cfg = json.load(fid)
    rnd_er = ana_cfg["analyses"]["simplex_counts"]["source"] =\
        os.path.join(TEST_DATA_DIR, ana_cfg["analyses"]["simplex_counts"]["source"])
    rnd_er = ana_cfg["analyses"]["simplex_counts"]["decorators"][1]["analysis_arg"]["random_er"]
    rnd_er["source"] = os.path.join(TEST_DATA_DIR, rnd_er["source"])
    analyses = list(test_module.get_analyses(ana_cfg))

    for analysis in analyses:
        result = analysis.apply(M, nrn).unstack("Control")
        nrm_result = (result["data"] - result["random_er"]) / result["random_er"]

        assert (nrm_result[:, 0] == 0).all()
        for word in RANDOM_WORDS:
            assert word in nrm_result
        assert nrm_result["orange"].index.name == "dim"
