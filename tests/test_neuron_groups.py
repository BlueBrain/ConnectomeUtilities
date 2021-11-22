import os
import pytest
import pandas

import bluepy

from conntility.circuit_models import neuron_groups as test_module

CIRC_FN = "/gpfs/bbp.cscs.ch/project/proj83/circuits/Bio_M/20200805/CircuitConfig_TC_WM"
CIRC = bluepy.Circuit(CIRC_FN)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_load_neurons():
    nrn = test_module.load_neurons(CIRC, ["x", "y", "z", "mtype", "etype"], base_target="Layer1")
    cmp = nrn["etype"].value_counts()
    reference = pandas.read_json(os.path.join(TEST_DATA_DIR, "reference_load_neurons.json"), orient="index")
    assert (reference[0] == cmp).all()
