from libsonata import SpikeReader
import pandas.testing as pdt
import pandas as pd
import numpy as np
import numpy.testing as npt
import pytest
from mock import patch

from bluepysnap.simulation import Simulation
import bluepysnap.spike_report as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.bbp import Cell

from utils import TEST_DATA_DIR


class TestSpikeReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SpikeReport(self.simulation)

    def test_config(self):
        assert self.test_obj.config == {"output_dir": str(TEST_DATA_DIR / "reporting"),
                                        "log_file": "log_spikes.log",
                                        "spikes_file": "spikes.h5",
                                        "spikes_sort_order": "time"}

    def test_t_start(self):
        assert self.test_obj.t_start == 0.

    def test_t_stop(self):
        assert self.test_obj.t_stop == 1000.

    def test_dt(self):
        assert self.test_obj.dt == 0.01

    def test_sim(self):
        assert isinstance(self.test_obj.sim, Simulation)

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

    def test_log(self):
        lines = ["Log for spikes.", "Second line."]
        with self.test_obj.log() as log:
            for l, line in enumerate(log):
                assert lines[l] in line

    def test_log2(self):
        simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        simulation._config["output"]["log_file"] = "unknown"
        test_obj = test_module.SpikeReport(simulation)
        with pytest.raises(BluepySnapError):
            with test_obj.log() as log:
                log.readlines()


class TestPopulationSpikeReport:
    def setup(self):
        self.simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
        self.test_obj = test_module.SpikeReport(self.simulation)["default"]

    def test_sorting(self):
        assert self.test_obj.sorting == "by_time"

    def test_name(self):
        assert self.test_obj.name == "default"

    def test_nodes(self):
        node_ids = self.test_obj.get([2], t_start=0.5).to_numpy()[0]
        assert self.test_obj.nodes.get(group=node_ids, properties=Cell.MTYPE) == "L6_Y"

    def test_nodes_2(self):
        test_obj = test_module.SpikeReport(self.simulation)["default2"]
        test_obj._population_name = "unknown"
        with pytest.raises(BluepySnapError):
            test_obj.nodes

    def test__resolve_nodes(self):
        npt.assert_array_equal(self.test_obj._resolve_nodes({Cell.MTYPE: "L6_Y"}), [1, 2])
        assert self.test_obj._resolve_nodes({Cell.MTYPE: "L2_X"}) == [0]
        npt.assert_array_equal(self.test_obj._resolve_nodes("Node12_L6_Y"), [1, 2])

    def test_get(self):
        pdt.assert_series_equal(self.test_obj.get(),
                                pd.Series([2, 0, 1, 2, 0], index=[0.1, 0.2, 0.3, 0.7, 1.3],
                                          name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get(2),
                                pd.Series([2, 2], index=[0.1, 0.7], name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2, 0]),
                                pd.Series([2, 0, 2, 0], index=[0.1, 0.2, 0.7, 1.3],
                                          name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([0, 2]),
                                pd.Series([2, 0, 2, 0], index=[0.1, 0.2, 0.7, 1.3],
                                          name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get(np.asarray([0, 2])),
                                pd.Series([2, 0, 2, 0], index=[0.1, 0.2, 0.7, 1.3],
                                          name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2], t_stop=0.5),
                                pd.Series([2], index=[0.1], name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2], t_start=0.5),
                                pd.Series([2], index=[0.7], name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2], t_start=0.5, t_stop=0.8),
                                pd.Series([2], index=[0.7], name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2, 1], t_start=0.5, t_stop=0.8),
                                pd.Series([2], index=[0.7], name="default_node_ids"))

        pdt.assert_series_equal(self.test_obj.get([2, 1], t_start=0.2, t_stop=0.8),
                                pd.Series([1, 2], index=[0.3, 0.7], name="default_node_ids"))

        # pdt.assert_series_equal(self.test_obj.get([0, 2], t_start=12), # TODO: FIXME PB SONATA
        #                         pd.Series([], index=[],
        #                                   name="default_node_ids"))

        pdt.assert_series_equal(
            self.test_obj.get(group={Cell.MTYPE: "L6_Y"}, t_start=0.2, t_stop=0.8),
            pd.Series([1, 2], index=[0.3, 0.7], name="default_node_ids"))

        pdt.assert_series_equal(
            self.test_obj.get(group={Cell.MTYPE: "L2_X"}),
            pd.Series([0, 0], index=[0.2, 1.3], name="default_node_ids"))

        pdt.assert_series_equal(
            self.test_obj.get(group="Layer23"),
            pd.Series([0, 0], index=[0.2, 1.3], name="default_node_ids"))

        with pytest.raises(BluepySnapError):
            self.test_obj.get(4)

    def test_get2(self):
        test_obj = test_module.SpikeReport(self.simulation)["default2"] # TODO: FIXME PB SONATA
        assert test_obj.sorting == "by_id"

        pdt.assert_series_equal(test_obj.get([2, 0]),
                                pd.Series([2, 0, 2, 0], index=[0.1, 0.2, 0.7, 1.3],
                                          name="default2_node_ids"))
        pdt.assert_series_equal(test_obj.get([0, 2]),
                                pd.Series([2, 0, 2, 0], index=[0.1, 0.2, 0.7, 1.3],
                                          name="default2_node_ids"))


    @patch(test_module.__name__ + '.PopulationSpikeReport._resolve_nodes',
           return_value=np.asarray([4]))
    def test_get3(self, mock):
        pdt.assert_series_equal(self.test_obj.get(4),
                                pd.Series([], index=[], name="default_node_ids"))

    def test_get_gid(self):
        npt.assert_allclose(self.test_obj.get_node_id(0), [0.2, 1.3])
        npt.assert_allclose(self.test_obj.get_node_id(0, t_start=1.), [1.3])
        npt.assert_allclose(self.test_obj.get_node_id(0, t_stop=1.), [0.2])
        npt.assert_allclose(self.test_obj.get_node_id(0, t_start=1., t_stop=12), [1.3])
        npt.assert_allclose(self.test_obj.get_node_id(0, t_start=0.1, t_stop=12), [0.2, 1.3])
