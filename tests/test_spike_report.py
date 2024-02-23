from unittest.mock import patch

import libsonata
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from libsonata import SpikeReader

import bluepysnap.spike_report as test_module
from bluepysnap.bbp import Cell
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.circuit_ids_types import IDS_DTYPE, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.simulation import Simulation

from utils import TEST_DATA_DIR, copy_test_data, edit_config

try:
    Output = libsonata._libsonata.Output
except AttributeError:
    from libsonata._libsonata import SimulationConfig

    Output = SimulationConfig.Output


def _create_series(node_ids, index, name="ids"):
    def _get_index(ids):
        return pd.Index(ids, name="times")

    if len(node_ids) == 0:
        # removing warning
        return pd.Series(node_ids, index=_get_index(index), name=name, dtype=np.int64)
    return pd.Series(node_ids, index=_get_index(index), name=name)


class TestSpikeReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.SpikeReport(self.simulation)

    def test_config(self):
        assert isinstance(self.test_obj.config, Output)
        assert self.test_obj.config.output_dir == str(TEST_DATA_DIR / "reporting")
        assert self.test_obj.config.log_file == "log_spikes.log"
        assert self.test_obj.config.spikes_file == "spikes.h5"
        assert self.test_obj.config.spikes_sort_order.name == "by_time"

    def test_time_start(self):
        assert self.test_obj.time_start == 0.0

    def test_time_stop(self):
        assert self.test_obj.time_stop == 1000.0

    def test_dt(self):
        assert self.test_obj.dt == 0.01

    def test_time_units(self):
        with pytest.raises(NotImplementedError):
            self.test_obj.time_units

    def test_simulation(self):
        assert isinstance(self.test_obj.simulation, Simulation)

    def test_log(self):
        lines = ["Log for spikes.", "Second line."]
        with self.test_obj.log() as log:
            for l, line in enumerate(log):
                assert lines[l] in line

    def test_log2(self):
        with copy_test_data(config="simulation_config.json") as (_, config_path):
            with edit_config(config_path) as config:
                config["output"]["log_file"] = "unknown"

            simulation = Simulation(config_path)
            test_obj = test_module.SpikeReport(simulation)
            with pytest.raises(BluepySnapError):
                with test_obj.log() as log:
                    log.readlines()

    def test___spike_reader(self):
        assert isinstance(self.test_obj._spike_reader, SpikeReader)

    def test_population_names(self):
        assert sorted(self.test_obj.population_names) == ["default", "default2"]

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.PopulationSpikeReport)

    def test_iter(self):
        assert sorted(list(self.test_obj)) == ["default", "default2"]
        for spikes in self.test_obj:
            isinstance(spikes, test_module.PopulationSpikeReport)

    def test_filter(self):
        filtered = self.test_obj.filter(group=None, t_start=0.3, t_stop=0.7)
        assert filtered.spike_report == self.test_obj
        assert filtered.t_start == 0.3
        assert filtered.t_stop == 0.7
        assert filtered.group is None
        assert isinstance(filtered, test_module.FilteredSpikeReport)
        npt.assert_allclose(filtered.report.index, np.array([0.3, 0.3, 0.7, 0.7]))
        assert filtered.report.columns.tolist() == ["ids", "population"]

        filtered = self.test_obj.filter(group={"other1": ["B"]}, t_start=0.3, t_stop=0.7)
        npt.assert_allclose(filtered.report.index, np.array([0.3]))
        assert filtered.report["population"].unique() == ["default2"]

        filtered = self.test_obj.filter(group={"population": "default2"})
        assert filtered.report["population"].unique() == ["default2"]
        npt.assert_array_equal(sorted(filtered.report["ids"].unique()), [0, 1, 2])

        filtered = self.test_obj.filter(group={"population": "default3"}, t_start=0.3, t_stop=0.6)
        assert len(filtered.report) == 0
        assert set(filtered.report.columns) == {"ids", "population"}
        assert filtered.report.index.name == "times"

        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 1, 2])
        filtered = self.test_obj.filter(group=ids)
        npt.assert_array_equal(sorted(filtered.report["ids"].unique()), [0, 1, 2])
        npt.assert_array_equal(
            filtered.report[filtered.report["population"] == "default"]["ids"].unique(), [0, 1]
        )

        filtered = self.test_obj.filter(group=0)
        npt.assert_array_equal(sorted(filtered.report["ids"].unique()), [0])
        npt.assert_array_equal(
            sorted(filtered.report["population"].unique()), ["default", "default2"]
        )

        filtered = self.test_obj.filter(group=[0, 1])
        npt.assert_array_equal(sorted(filtered.report["ids"].unique()), [0, 1])
        npt.assert_array_equal(
            sorted(filtered.report["population"].unique()), ["default", "default2"]
        )


class TestPopulationSpikeReport:
    def setup_method(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
        self.test_obj = test_module.SpikeReport(self.simulation)["default"]

    def test__sorted_by(self):
        assert self.test_obj._sorted_by == "by_time"

    def test_name(self):
        assert self.test_obj.name == "default"

    def test_nodes(self):
        node_ids = self.test_obj.get([2], t_start=0.5).to_numpy()[0]
        assert self.test_obj.nodes.get(group=node_ids, properties=Cell.MTYPE) == "L6_Y"
        npt.assert_equal(node_ids.dtype, IDS_DTYPE)

    def test_nodes_invalid_population(self):
        test_obj = test_module.SpikeReport(self.simulation)["default2"]
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.nodes

    def test__resolve_nodes(self):
        npt.assert_array_equal(self.test_obj.resolve_nodes({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj.resolve_nodes({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj.resolve_nodes("Node12_L6_Y"), [1, 2])

    def test_get(self):
        tested = self.test_obj.get()
        pdt.assert_series_equal(tested, _create_series([2, 0, 1, 2, 0], [0.1, 0.2, 0.3, 0.7, 1.3]))
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        pdt.assert_series_equal(self.test_obj.get([]), _create_series([], []))
        pdt.assert_series_equal(self.test_obj.get(np.array([])), _create_series([], []))
        pdt.assert_series_equal(self.test_obj.get(()), _create_series([], []))
        pdt.assert_series_equal(self.test_obj.get(2), _create_series([2, 2], [0.1, 0.7]))

        pdt.assert_series_equal(
            self.test_obj.get(CircuitNodeId("default", 2)), _create_series([2, 2], [0.1, 0.7])
        )
        # not in population
        pdt.assert_series_equal(
            self.test_obj.get(CircuitNodeId("default2", 2)), _create_series([], [])
        )

        pdt.assert_series_equal(self.test_obj.get(0, t_start=1.0), _create_series([0], [1.3]))
        pdt.assert_series_equal(self.test_obj.get(0, t_stop=1.0), _create_series([0], [0.2]))
        pdt.assert_series_equal(
            self.test_obj.get(0, t_start=1.0, t_stop=12), _create_series([0], [1.3])
        )
        pdt.assert_series_equal(
            self.test_obj.get(0, t_start=0.1, t_stop=12), _create_series([0, 0], [0.2, 1.3])
        )

        pdt.assert_series_equal(
            self.test_obj.get([2, 0]), _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3])
        )

        pdt.assert_series_equal(
            self.test_obj.get([0, 2]), _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3])
        )

        pdt.assert_series_equal(
            self.test_obj.get(np.asarray([0, 2])),
            _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3]),
        )

        pdt.assert_series_equal(self.test_obj.get([2], t_stop=0.5), _create_series([2], [0.1]))

        pdt.assert_series_equal(self.test_obj.get([2], t_start=0.5), _create_series([2], [0.7]))

        pdt.assert_series_equal(
            self.test_obj.get([2], t_start=0.5, t_stop=0.8), _create_series([2], [0.7])
        )

        pdt.assert_series_equal(
            self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8), _create_series([2], [0.7])
        )

        pdt.assert_series_equal(
            self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8), _create_series([1, 2], [0.3, 0.7])
        )

        pdt.assert_series_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8),
            _create_series([1, 2], [0.3, 0.7]),
        )

        pdt.assert_series_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}), _create_series([0, 0], [0.2, 1.3])
        )

        pdt.assert_series_equal(
            self.test_obj.get(group="Layer23"), _create_series([0, 0], [0.2, 1.3])
        )

        # test that simulation node_set is used
        pdt.assert_series_equal(
            self.test_obj.get("only_exists_in_simulation"),
            _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3]),
        )

        # no 0.1, 0.7 from  ("default2", 2)
        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 1, 2])
        npt.assert_array_equal(self.test_obj.get(ids), _create_series([0, 1, 0], [0.2, 0.3, 1.3]))

        with pytest.raises(BluepySnapError):
            self.test_obj.get([-1], t_start=0.2)

        with pytest.raises(BluepySnapError):
            self.test_obj.get([0, 2], t_start=-1)

        with pytest.raises(BluepySnapError):
            self.test_obj.get([0, 2], t_start=12)

        with pytest.raises(BluepySnapError):
            self.test_obj.get(4)

    def test_get2(self):
        test_obj = test_module.SpikeReport(self.simulation)["default2"]
        assert test_obj._sorted_by == "by_id"
        pdt.assert_series_equal(
            test_obj.get([2, 0]), _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3], name="ids")
        )

        pdt.assert_series_equal(
            test_obj.get([0, 2]), _create_series([2, 0, 2, 0], [0.1, 0.2, 0.7, 1.3], name="ids")
        )

    @patch(
        test_module.__name__ + ".PopulationSpikeReport.resolve_nodes", return_value=np.asarray([4])
    )
    def test_get_not_in_report(self, mock):
        pdt.assert_series_equal(self.test_obj.get(4), _create_series([], []))

    @patch(
        test_module.__name__ + ".PopulationSpikeReport.resolve_nodes",
        return_value=np.asarray([0, 4]),
    )
    def test_get_not_in_report(self, mock):
        pdt.assert_series_equal(self.test_obj.get([0, 4]), _create_series([0, 0], [0.2, 1.3]))

    def test_node_ids(self):
        npt.assert_array_equal(self.test_obj.node_ids, np.array(sorted([0, 1, 2]), dtype=np.int64))
