# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import pandas

import bluepysnap as snap

from conntility.circuit_models import neuron_groups as test_module

from utils import get_snap_test_circuit

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def test_load_neurons():
    with get_snap_test_circuit() as circ_fn:
        CIRC = snap.Circuit(circ_fn)
        nrn = test_module.load_neurons(CIRC, ["x", "y", "z", "@dynamics:holding_current"])
        cmp = nrn["@dynamics:holding_current"].value_counts()
        reference = pandas.read_csv(os.path.join(TEST_DATA_DIR, "reference_load_neurons.csv"),
                                     index_col=0)
        assert (reference["count"] == cmp).all()


def test_load_config_and_grouping():
    with get_snap_test_circuit() as circ_fn:
        CIRC = snap.Circuit(circ_fn)
        load_cfg = os.path.join(TEST_DATA_DIR, "test_load_config.json")
        nrn = test_module.load_group_filter(CIRC, load_cfg)

        assert len(nrn.index.names) == 3
        assert len(nrn) == 3
        assert nrn["grid-subtarget"].value_counts()[0] == 2
        assert nrn["grid-subtarget"].value_counts()[1] == 1
        assert nrn.index.to_frame()["grid-i"].min() == -1
        assert nrn["grid-y"].apply(lambda v: v == pytest.approx(225)).all()
