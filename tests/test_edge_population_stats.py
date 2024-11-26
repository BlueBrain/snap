from unittest.mock import Mock

import numpy.testing as npt
import pytest

import bluepysnap.edges.edge_population_stats as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.circuit_ids import CircuitNodeId


class TestStatsHelper:
    def setup_method(self):
        self.edge_pop = Mock()
        self.stats = test_module.StatsHelper(self.edge_pop)

    def test_divergence_by_synapses(self):
        self.edge_pop.source.ids.return_value = [1, 2]
        self.edge_pop.iter_connections.return_value = [
            (CircuitNodeId(population="default", id=1), None, 42),
            (CircuitNodeId(population="default", id=1), None, 43),
        ]
        actual = self.stats.divergence("pre", "post", by="synapses")
        npt.assert_equal(actual, [85, 0])

    def test_divergence_by_connections(self):
        self.edge_pop.source.ids.return_value = [1, 2]
        self.edge_pop.iter_connections.return_value = [
            (CircuitNodeId(population="default", id=1), None),
            (CircuitNodeId(population="default", id=1), None),
        ]
        actual = self.stats.divergence("pre", "post", by="connections")
        npt.assert_equal(actual, [2, 0])

    def test_divergence_error(self):
        pytest.raises(BluepySnapError, self.stats.divergence, "pre", "post", by="err")

    def test_convergence_by_synapses(self):
        self.edge_pop.target.ids.return_value = [1, 2]
        self.edge_pop.iter_connections.return_value = [
            (None, CircuitNodeId(population="default", id=2), 42),
            (None, CircuitNodeId(population="default", id=2), 43),
        ]
        actual = self.stats.convergence("pre", "post", by="synapses")
        npt.assert_equal(actual, [0, 85])

    def test_convergence_by_connections(self):
        self.edge_pop.target.ids.return_value = [1, 2]
        self.edge_pop.iter_connections.return_value = [
            (None, CircuitNodeId(population="default", id=2)),
            (None, CircuitNodeId(population="default", id=2)),
        ]
        actual = self.stats.convergence("pre", "post", by="connections")
        npt.assert_equal(actual, [0, 2])

    def test_convergence_error(self):
        pytest.raises(BluepySnapError, self.stats.convergence, "pre", "post", by="err")
