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
from bluepysnap.circuit_ids import CircuitNodeIds, CircuitNodeId

from utils import TEST_DATA_DIR


class TestFrameReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.FrameReport(self.simulation, "soma_report")
        self.test_obj_info = test_module.FrameReport(self.simulation, "section_report")

    def test_config(self):
        assert self.test_obj.config == {"cells": "Layer23", "variable_name": "m",
                                        "sections": "soma", "enabled": True}

    def test_time_start(self):
        assert self.test_obj.time_start == 0.
        assert self.test_obj_info.time_start == 0.2

    def test_time_stop(self):
        assert self.test_obj.time_stop == 1000.
        assert self.test_obj_info.time_stop == 0.8

    def test_dt(self):
        assert self.test_obj.dt == 0.01
        with patch('bluepysnap.frame_report.L') as log_mock:
            assert self.test_obj_info.dt == 0.02
            assert log_mock.warning.call_count == 1

    def test_time_units(self):
        assert self.test_obj.time_units == "ms"
        with pytest.raises(BluepySnapError):
            self.test_obj_info.time_units

    def test_data_units(self):
        assert self.test_obj.data_units == "mV"
        with pytest.raises(BluepySnapError):
            self.test_obj_info.data_units

    def test_sim(self):
        assert isinstance(self.test_obj.simulation, Simulation)
        assert isinstance(self.test_obj_info.simulation, Simulation)

    def test_node_set(self):
        assert self.test_obj.node_set == {"layer": [2, 3]}
        assert self.test_obj_info.node_set == {"layer": [2, 3]}

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default", "default2"]
        assert sorted(self.test_obj_info.population_names) == ["default", "default2"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationFrameReport)
        assert isinstance(self.test_obj_info["default"], test_module.PopulationFrameReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationFrameReport)

        assert sorted(list(self.test_obj_info)) == ["default", "default2"]
        for report in self.test_obj_info:
            isinstance(report, test_module.PopulationFrameReport)


class TestCompartmentReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.CompartmentReport(self.simulation, "section_report")

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationCompartmentReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationCompartmentReport)

    def test_filter(self):
        filtered = self.test_obj.filter(group=[0], t_start=0.3, t_stop=0.6)
        assert filtered.frame_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.6
        assert filtered.group == [0]
        assert isinstance(filtered, test_module.FilteredFrameReport)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [("default", 0, 0), ("default", 0, 1),
                                                    ("default2", 0, 0), ("default2", 0, 1)]

        filtered = self.test_obj.filter(group={"other1": ["B"]}, t_start=0.3, t_stop=0.6)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [("default2", 1, 0), ("default2", 1, 1)]

        filtered = self.test_obj.filter(group={"population": "default2"}, t_start=0.3, t_stop=0.6)
        assert filtered.report.columns.tolist() == [("default2", 0, 0), ("default2", 0, 1),
                                                    ("default2", 1, 0), ("default2", 1, 1),
                                                    ("default2", 2, 0), ("default2", 2, 1),
                                                    ("default2", 2, 1)]

        filtered = self.test_obj.filter(group={"population": "default3"}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, pd.DataFrame())


class TestSomaReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SomaReport(self.simulation, "soma_report")

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationSomaReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationSomaReport)

    def test_filter(self):
        filtered = self.test_obj.filter(group=None, t_start=0.3, t_stop=0.6)
        assert filtered.frame_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.6
        assert filtered.group is None
        assert isinstance(filtered, test_module.FilteredFrameReport)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [("default", 0), ("default", 1), ("default", 2),
                                                    ("default2", 0), ("default2", 1),
                                                    ("default2", 2)]

        filtered = self.test_obj.filter(group={"other1": ["B"]}, t_start=0.3, t_stop=0.6)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.4, 0.5, 0.6]))
        assert filtered.report.columns.tolist() == [("default2", 1)]

        filtered = self.test_obj.filter(group={"population": "default2"}, t_start=0.3, t_stop=0.6)
        assert filtered.report.columns.tolist() == [("default2", 0), ("default2", 1),
                                                    ("default2", 2)]

        filtered = self.test_obj.filter(group={"population": "default3"}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, pd.DataFrame())

        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 1, 1])
        filtered = self.test_obj.filter(group=ids, t_start=0.3, t_stop=0.6)
        assert filtered.report.columns.tolist() == [("default", 0), ("default", 1), ("default2", 1)]
        ids = CircuitNodeIds.from_tuples([("default2", 1)])
        npt.assert_allclose(filtered.report.loc[:, ids.index].index, np.array([0.3, 0.4, 0.5, 0.6]))
        npt.assert_allclose(filtered.report.loc[:, ids.index], np.array([[1.3, 1.4, 1.5, 1.6]]).T)


class TestPopulationFrameReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.FrameReport(self.simulation, "section_report")["default"]

    def test_name(self):
        assert self.test_obj.name == "default"

    def test__resolve(self):
        with pytest.raises(NotImplementedError):
            self.test_obj._resolve([1])


class TestPopulationCompartmentReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.CompartmentReport(self.simulation, "section_report")["default"]
        timestamps = np.linspace(0, 0.9, 10)
        data = np.array([np.arange(7) + j * 0.1 for j in range(10)], dtype=np.float32)
        ids = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 1)]
        self.df = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(ids), index=timestamps)

    def test__resolve(self):
        npt.assert_array_equal(self.test_obj._resolve({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj._resolve({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj._resolve("Node12_L6_Y"), [1, 2])

    def test_nodes(self):
        assert self.test_obj.nodes.get(group=2, properties=Cell.MTYPE) == "L6_Y"

    def test_nodes_invalid_population(self):
        test_obj = self.test_obj.__class__(self.test_obj.frame_report, self.test_obj.name)
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.nodes

    def test_get(self):
        pdt.assert_frame_equal(self.test_obj.get(), self.df)

        pdt.assert_frame_equal(self.test_obj.get([]), pd.DataFrame())
        pdt.assert_frame_equal(self.test_obj.get(np.array([])), pd.DataFrame())
        pdt.assert_frame_equal(self.test_obj.get(()), pd.DataFrame())

        pdt.assert_frame_equal(self.test_obj.get(2), self.df.loc[:, [2]])
        pdt.assert_frame_equal(self.test_obj.get(CircuitNodeId("default", 2)), self.df.loc[:, [2]])

        # not from this population
        pdt.assert_frame_equal(self.test_obj.get(CircuitNodeId("default2", 2)),  pd.DataFrame())

        pdt.assert_frame_equal(self.test_obj.get([2, 0]), self.df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([0, 2]), self.df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get(np.asarray([0, 2])), self.df.loc[:, [0, 2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.5), self.df.iloc[:6].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_stop=0.55), self.df.iloc[:6].loc[:, [2]])

        pdt.assert_frame_equal(self.test_obj.get([2], t_start=0.5), self.df.iloc[5:].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8), self.df.iloc[5:9].loc[:, [2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8), self.df.iloc[5:9].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8), self.df.iloc[2:9].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8),
            self.df.iloc[2:9].loc[:, [1, 2]])

        pdt.assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}), self.df.loc[:, [0]])

        pdt.assert_frame_equal(
            self.test_obj.get(group="Layer23"), self.df.loc[:, [0]])

        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 2, 1])
        pdt.assert_frame_equal(self.test_obj.get(group=ids), self.df.loc[:, [0, 2]])

        with pytest.raises(BluepySnapError):
            self.test_obj.get(-1, t_start=0.2)

        with pytest.raises(BluepySnapError):
            self.test_obj.get(0, t_start=-1)

        with pytest.raises(BluepySnapError):
            self.test_obj.get([0, 2], t_start=15)

        with pytest.raises(BluepySnapError):
            self.test_obj.get(4)

    def test_get_partially_not_in_report(self):
        with patch.object(self.test_obj.__class__, "_resolve", return_value=np.asarray([0, 4])):
            pdt.assert_frame_equal(self.test_obj.get([0, 4]),  self.df.loc[:, [0]])

    def test_get_not_in_report(self):
        with patch.object(self.test_obj.__class__, "_resolve", return_value=np.asarray([4])):
            pdt.assert_frame_equal(self.test_obj.get([4]), pd.DataFrame())

    def test_node_ids(self):
        npt.assert_array_equal(self.test_obj.node_ids, np.array(sorted([0, 1, 2]), dtype=np.int64))


class TestPopulationSomaReport(TestPopulationCompartmentReport):
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SomaReport(self.simulation, "soma_report")["default"]
        timestamps = np.linspace(0, 0.9, 10)
        data = {0: timestamps, 1: timestamps + 1, 2: timestamps + 2}
        self.df = pd.DataFrame(data=data, index=timestamps, columns=[0, 1, 2]).astype(np.float32)
