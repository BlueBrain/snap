from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepysnap.frame_report as test_module
from bluepysnap.bbp import Cell
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.circuit_ids_types import IDS_DTYPE, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.simulation import Simulation

from utils import TEST_DATA_DIR


class TestFrameReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.FrameReport(self.simulation, "soma_report")
        self.test_obj_info = test_module.FrameReport(self.simulation, "section_report")

    def test_config(self):
        assert self.test_obj.config == {
            "cells": "Layer23",
            "variable_name": "m",
            "sections": "soma",
            "type": "compartment",
            "file_name": "soma_report",
            "start_time": 0,
            "end_time": 1000.0,
            "dt": 0.01,
            "enabled": True,
        }

    def test_time_start(self):
        assert self.test_obj.time_start == 0.0
        assert self.test_obj_info.time_start == 0.2

    def test_time_stop(self):
        assert self.test_obj.time_stop == 1000.0
        assert self.test_obj_info.time_stop == 0.8

    def test_dt(self):
        assert self.test_obj.dt == 0.01
        with patch("bluepysnap.frame_report.L") as log_mock:
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
        assert self.test_obj.node_set == "Layer23"
        assert self.test_obj_info.node_set == "Layer23"

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
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.CompartmentReport(self.simulation, "section_report")

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationCompartmentReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationCompartmentReport)

    def test_filter(self):
        expected = pd.DataFrame(
            data=[
                np.array([0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3] * 2, dtype=np.float32) + 0.1 * i
                for i in range(4)
            ],
            columns=pd.MultiIndex.from_tuples(
                [
                    ("default", 0, 0),
                    ("default", 0, 1),
                    ("default", 1, 0),
                    ("default", 1, 1),
                    ("default", 2, 0),
                    ("default", 2, 1),
                    ("default", 2, 1),
                    ("default2", 0, 0),
                    ("default2", 0, 1),
                    ("default2", 1, 0),
                    ("default2", 1, 1),
                    ("default2", 2, 0),
                    ("default2", 2, 1),
                    ("default2", 2, 1),
                ]
            ),
            index=np.array([0.3, 0.4, 0.5, 0.6]),
        )

        filtered = self.test_obj.filter(group=[0], t_start=0.3, t_stop=0.6)
        assert filtered.frame_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.6
        assert filtered.group == [0]
        assert isinstance(filtered, test_module.FilteredFrameReport)
        expected_columns = [
            ("default", 0, 0),
            ("default", 0, 1),
            ("default2", 0, 0),
            ("default2", 0, 1),
        ]
        pdt.assert_frame_equal(filtered.report, expected.loc[:, expected_columns])

        filtered = self.test_obj.filter(group={"other1": ["B"]}, t_start=0.3, t_stop=0.6)
        expected_columns = [("default2", 1, 0), ("default2", 1, 1)]
        pdt.assert_frame_equal(filtered.report, expected.loc[:, expected_columns])

        filtered = self.test_obj.filter(group={"population": "default2"}, t_start=0.3, t_stop=0.6)
        expected_columns = [
            ("default2", 0, 0),
            ("default2", 0, 1),
            ("default2", 1, 0),
            ("default2", 1, 1),
            ("default2", 2, 0),
            ("default2", 2, 1),
        ]
        pdt.assert_frame_equal(filtered.report, expected.loc[:, expected_columns])

        filtered = self.test_obj.filter(group={"population": "default3"}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, expected.iloc[:0, :0])


class TestSomaReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.SomaReport(self.simulation, "soma_report")

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationSomaReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for report in self.test_obj:
            isinstance(report, test_module.PopulationSomaReport)

    def test_filter(self):
        expected = pd.DataFrame(
            data=[
                np.array([0.3, 1.3, 2.3, 0.3, 1.3, 2.3], dtype=np.float32) + 0.1 * i
                for i in range(4)
            ],
            columns=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                ]
            ),
            index=np.array([0.3, 0.4, 0.5, 0.6]),
        )

        filtered = self.test_obj.filter(group=None, t_start=0.3, t_stop=0.6)
        assert filtered.frame_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.6
        assert filtered.group is None
        assert isinstance(filtered, test_module.FilteredFrameReport)
        pdt.assert_frame_equal(filtered.report, expected)

        filtered = self.test_obj.filter(group={"other1": ["B"]}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, expected.loc[:, [("default2", 1)]])

        filtered = self.test_obj.filter(group={"population": "default2"}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, expected.loc[:, ["default2"]])

        filtered = self.test_obj.filter(group={"population": "default3"}, t_start=0.3, t_stop=0.6)
        pdt.assert_frame_equal(filtered.report, expected.iloc[:0, :0])

        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 1, 1])
        filtered = self.test_obj.filter(group=ids, t_start=0.3, t_stop=0.6)
        expected_columns = [("default", 0), ("default", 1), ("default2", 1)]
        pdt.assert_frame_equal(filtered.report, expected.loc[:, expected_columns])


class TestPopulationFrameReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.FrameReport(self.simulation, "section_report")["default"]

    def test_name(self):
        assert self.test_obj.name == "default"

    def test__resolve(self):
        with pytest.raises(NotImplementedError):
            self.test_obj.resolve_nodes([1])


class TestPopulationCompartmentReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.CompartmentReport(self.simulation, "section_report")["default"]
        timestamps = np.linspace(0, 0.9, 10)
        data = np.array([np.arange(7) + j * 0.1 for j in range(10)], dtype=np.float32)
        ids = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 1)]
        self.df = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(ids), index=timestamps)

    @property
    def empty_df(self):
        """Return an empty DataFrame with the original types of index and columns."""
        return self.df.iloc[:0, :0]

    def test__resolve(self):
        npt.assert_array_equal(self.test_obj.resolve_nodes({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj.resolve_nodes({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj.resolve_nodes("Node12_L6_Y"), [1, 2])

    def test_nodes(self):
        assert self.test_obj.nodes.get(group=2, properties=Cell.MTYPE) == "L6_Y"

    def test_nodes_invalid_population(self):
        test_obj = self.test_obj.__class__(self.test_obj.frame_report, self.test_obj.name)
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.nodes

    @pytest.mark.parametrize("t_step", [None, 0.02, 0.04, 0.0401, 0.0399, 0.05, 200000])
    def test_get(self, t_step):
        def _assert_frame_equal(df1, df2):
            # compare df1 and df2, after filtering df2 according to t_stride
            df2 = df2.iloc[::t_stride]
            pdt.assert_frame_equal(df1, df2)

        # calculate the expected t_stride, depending on t_step and dt (varying across tests)
        t_stride = round(t_step / self.test_obj.frame_report.dt) if t_step is not None else 1

        _assert_frame_equal(self.test_obj.get(t_step=t_step), self.df)
        _assert_frame_equal(self.test_obj.get([], t_step=t_step), self.empty_df)
        _assert_frame_equal(self.test_obj.get(np.array([]), t_step=t_step), self.empty_df)
        _assert_frame_equal(self.test_obj.get((), t_step=t_step), self.empty_df)

        _assert_frame_equal(self.test_obj.get(2, t_step=t_step), self.df.loc[:, [2]])
        _assert_frame_equal(
            self.test_obj.get(CircuitNodeId("default", 2), t_step=t_step), self.df.loc[:, [2]]
        )

        # not from this population
        _assert_frame_equal(
            self.test_obj.get(CircuitNodeId("default2", 2), t_step=t_step), self.empty_df
        )

        _assert_frame_equal(self.test_obj.get([2, 0], t_step=t_step), self.df.loc[:, [0, 2]])

        _assert_frame_equal(self.test_obj.get([0, 2], t_step=t_step), self.df.loc[:, [0, 2]])

        _assert_frame_equal(
            self.test_obj.get(np.asarray([0, 2]), t_step=t_step), self.df.loc[:, [0, 2]]
        )

        _assert_frame_equal(
            self.test_obj.get([2], t_stop=0.5, t_step=t_step), self.df.iloc[:6].loc[:, [2]]
        )

        _assert_frame_equal(
            self.test_obj.get([2], t_stop=0.55, t_step=t_step), self.df.iloc[:6].loc[:, [2]]
        )

        _assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_step=t_step), self.df.iloc[5:].loc[:, [2]]
        )

        _assert_frame_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8, t_step=t_step),
            self.df.iloc[5:9].loc[:, [2]],
        )

        _assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8, t_step=t_step),
            self.df.iloc[5:9].loc[:, [1, 2]],
        )

        _assert_frame_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8, t_step=t_step),
            self.df.iloc[2:9].loc[:, [1, 2]],
        )

        _assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8, t_step=t_step),
            self.df.iloc[2:9].loc[:, [1, 2]],
        )

        _assert_frame_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}, t_step=t_step), self.df.loc[:, [0]]
        )

        _assert_frame_equal(self.test_obj.get(group="Layer23", t_step=t_step), self.df.loc[:, [0]])

        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 2, 1])
        _assert_frame_equal(self.test_obj.get(group=ids, t_step=t_step), self.df.loc[:, [0, 2]])

        # test that simulation node_set is used
        _assert_frame_equal(
            self.test_obj.get("only_exists_in_simulation", t_step=t_step), self.df.loc[:, [0, 2]]
        )

        with pytest.raises(
            BluepySnapError, match="All node IDs must be >= 0 and < 3 for population 'default'"
        ):
            self.test_obj.get(-1, t_start=0.2, t_step=t_step)

        with pytest.raises(BluepySnapError, match="Times cannot be negative"):
            self.test_obj.get(0, t_start=-1, t_step=t_step)

        with pytest.raises(BluepySnapError, match="tstart is after the end of the range"):
            self.test_obj.get([0, 2], t_start=15, t_step=t_step)

        with pytest.raises(
            BluepySnapError, match="All node IDs must be >= 0 and < 3 for population 'default'"
        ):
            self.test_obj.get(4, t_step=t_step)

    @pytest.mark.parametrize("t_step", [0, -1, 0.0000001])
    def test_get_with_invalid_t_step(self, t_step):
        match = f"Invalid t_step={t_step}. It should be None or a multiple of"
        with pytest.raises(BluepySnapError, match=match):
            self.test_obj.get(t_step=t_step)

    def test_get_partially_not_in_report(self):
        with patch.object(
            self.test_obj.__class__, "resolve_nodes", return_value=np.asarray([0, 4])
        ):
            pdt.assert_frame_equal(self.test_obj.get([0, 4]), self.df.loc[:, [0]])

    def test_get_not_in_report(self):
        with patch.object(self.test_obj.__class__, "resolve_nodes", return_value=np.asarray([4])):
            pdt.assert_frame_equal(self.test_obj.get([4]), self.empty_df)

    def test_node_ids(self):
        npt.assert_array_equal(self.test_obj.node_ids, np.array(sorted([0, 1, 2]), dtype=IDS_DTYPE))


class TestPopulationSomaReport(TestPopulationCompartmentReport):
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.SomaReport(self.simulation, "soma_report")["default"]
        timestamps = np.linspace(0, 0.9, 10)
        data = {0: timestamps, 1: timestamps + 1, 2: timestamps + 2}
        self.df = pd.DataFrame(data=data, index=timestamps, columns=[0, 1, 2]).astype(np.float32)
