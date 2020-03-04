import pandas.testing as pdt
import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest

from bluepysnap.simulation import Simulation
import bluepysnap.frame_report as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.bbp import Cell

from utils import TEST_DATA_DIR


class TestFrameReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.FrameReport(self.simulation, "soma_report")
        self.test_obj_info = test_module.FrameReport(self.simulation, "soma_report2")

    def test_config(self):
        assert self.test_obj.config == {"cells": "Layer23", "variable_name": "m",
                                        "sections": "soma", "enabled": True}

    def test_t_start(self):
        assert self.test_obj.t_start == 0.
        assert self.test_obj_info.t_start == 0.2

    def test_t_stop(self):
        assert self.test_obj.t_stop == 1000.
        assert self.test_obj_info.t_stop == 0.8

    def test_dt(self):
        assert self.test_obj.dt == 0.01
        assert self.test_obj_info.dt == 0.02

    def test_sim(self):
        assert isinstance(self.test_obj.sim, Simulation)

    def test_node_set(self):
        assert self.test_obj.node_set == {"layer": [2, 3]}

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default", "default2"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationFrameReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for spikes in self.test_obj:
            isinstance(spikes, test_module.PopulationFrameReport)


class TestPopulationFrameReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.FrameReport(self.simulation, "soma_report")["default"]

    def test_name(self):
        assert self.test_obj.name == "default"

    def test_sorted(self):
        assert self.test_obj.sorted

    def test_times(self):
        assert self.test_obj.times == (0.0, 1.0, 0.1)

    def test_time_units(self):
        assert self.test_obj.time_units == "ms"

    def test_data_units(self):
        assert self.test_obj.data_units == "mV"

    def test__resolve_nodes(self):
        npt.assert_array_equal(self.test_obj._resolve_nodes({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj._resolve_nodes({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj._resolve_nodes("Node12_L6_Y"), [1, 2])

    def test_nodes(self):
        assert self.test_obj.nodes.get(group=2, properties=Cell.MTYPE) == "L6_Y"

    def test_nodes_2(self):
        test_obj = test_module.FrameReport(self.simulation, "soma_report")["default2"]
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.nodes

    def test_get(self):
        # pdt.assert_frame_equal(self.test_obj.get(),)
        assert True

