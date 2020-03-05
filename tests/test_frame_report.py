import pandas.testing as pdt
import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest
from mock import patch

from bluepysnap.simulation import Simulation
import bluepysnap.frame_report as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.bbp import Cell

from utils import TEST_DATA_DIR


class TestFrameReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.FrameReport(self.simulation, "soma_report")
        self.test_obj_info = test_module.FrameReport(self.simulation, "section_report")

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


class TestSomasReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SomasReport(self.simulation, "soma_report")

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default", "default2"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationSomasReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationSomasReport)


class TestCompartmentsReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.CompartmentsReport(self.simulation, "section_report")

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationCompartmentsReport)

    def test_iter(self):
        assert list(self.test_obj) == ["default"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationCompartmentsReport)


class TestPopulationSomasReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SomasReport(self.simulation, "soma_report")["default"]

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

    def test__resolve(self):
        npt.assert_array_equal(self.test_obj._resolve({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj._resolve({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj._resolve("Node12_L6_Y"), [1, 2])

    def test_population(self):
        assert self.test_obj.population.get(group=2, properties=Cell.MTYPE) == "L6_Y"

    def test_population_2(self):
        test_obj = test_module.SomasReport(self.simulation, "soma_report")["default2"]
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.population

    def test_get(self):
        values = np.linspace(0, 0.9, 10)
        data = {0: values, 1: values + 1, 2: values + 2}
        df = pd.DataFrame(data=data, index=values)
        pdt.assert_frame_equal(self.test_obj.get(), df)

        pdt.assert_frame_equal(self.test_obj.get(2), df.loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2, 0]), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([0, 2]), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get(np.asarray([0, 2])), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.5), df.iloc[:5].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.55), df.iloc[:5].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_start=0.5), df.iloc[5:].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8), df.iloc[5:8].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8),  df.iloc[5:8].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8), df.iloc[2:8].loc[:, [1, 2]])

        assert self.test_obj.get([0, 2], t_start=15).empty

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8), df.iloc[2:8].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}), df.loc[:, [0]])

        pdt.assert_frame_equal(
            self.test_obj.get(group="Layer23"), df.loc[:, [0]])

        with pytest.raises(BluepySnapError):
            self.test_obj.get(4)

    @patch(test_module.__name__ + '.PopulationFrameReport._resolve',
           return_value=np.asarray([4]))
    def test_get2(self, mock):
        pdt.assert_frame_equal(self.test_obj.get(4), pd.DataFrame())


class TestPopulationCompartmentsReport(TestPopulationSomasReport):
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.CompartmentsReport(self.simulation, "section_report")["default"]

    def test_times(self):
        assert self.test_obj.times == (0.0, 1.0, 0.1)

    def test__resolve(self):
        npt.assert_array_equal(self.test_obj._resolve({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj._resolve({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj._resolve("Node12_L6_Y"), [1, 2])

    def test_population(self):
        assert self.test_obj.population.get(group=2, properties=Cell.MTYPE) == "L6_Y"

    def test_population_2(self):
        test_obj = test_module.CompartmentsReport(self.simulation, "section_report")["default"]
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.population

    def test_get(self):
        values = np.linspace(0, 0.9, 10)
        data = np.array([np.arange(6) + j * 0.1 for j in range(10)])

        data = {(0, 0): data[:, 0], (0, 1): data[:, 1], (1, 0): data[:, 2], (1, 1): data[:, 3], (2, 0): data[:, 4], (2, 1): data[:, 5]}
        df = pd.DataFrame(data=data, index=values)

        pdt.assert_frame_equal(self.test_obj.get(), df)

        pdt.assert_frame_equal(self.test_obj.get(2), df.loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2, 0]), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([0, 2]), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get(np.asarray([0, 2])), df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.5), df.iloc[:5].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.55), df.iloc[:5].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_start=0.5), df.iloc[5:].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8), df.iloc[5:8].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8),  df.iloc[5:8].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8), df.iloc[2:8].loc[:, [1, 2]])

        assert self.test_obj.get([0, 2], t_start=15).empty

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8), df.iloc[2:8].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}), df.loc[:, [0]])

        pdt.assert_frame_equal(
            self.test_obj.get(group="Layer23"), df.loc[:, [0]])

        with pytest.raises(BluepySnapError):
            self.test_obj.get(4)

    @patch(test_module.__name__ + '.PopulationFrameReport._resolve',
           return_value=np.asarray([4]))
    def test_get2(self, mock):
        pdt.assert_frame_equal(self.test_obj.get(4), pd.DataFrame())
