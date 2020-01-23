import os

import mock
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

import libsonata
from mock import Mock

from bluepysnap.bbp import Synapse
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Edge

import bluepysnap.edges as test_module

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


def index_as_uint64(values):
    '''have pandas index types match'''
    return np.array(values, dtype=np.uint64)


def test_estimate_range_size_1():
    func = lambda x: Mock(ranges=np.zeros(x))
    actual = test_module._estimate_range_size(func, [11, 21, 31], n=5)
    npt.assert_equal(actual, 21)


def test_estimate_range_size_2():
    func = lambda x: Mock(ranges=[42])
    actual = test_module._estimate_range_size(func, range(10))
    npt.assert_equal(actual, 1)


def test_estimate_range_size_3():
    func = lambda x: Mock(ranges=[42])
    actual = test_module._estimate_range_size(func, range(10))
    npt.assert_equal(actual, 1)


def test_estimate_range_size_4():
    with pytest.raises(AssertionError):
        test_module._estimate_range_size(mock.ANY, [])


class TestEdgeStorage:
    def setup(self):
        config = {
            'edges_file': os.path.join(TEST_DATA_DIR, 'edges.h5'),
            'edge_types_file': None,
        }
        self.circuit = Mock()
        self.test_obj = test_module.EdgeStorage(config, self.circuit)

    def test_storage(self):
        assert isinstance(self.test_obj.storage, libsonata.EdgeStorage)

    def test_population_names(self):
        assert sorted(list(self.test_obj.population_names)) == ["default"]

    def test_circuit(self):
        assert self.test_obj.circuit is self.circuit

    def test_population(self):
        pop = self.test_obj.population("default")
        assert isinstance(pop, test_module.EdgePopulation)
        assert pop.name == "default"
        pop2 = self.test_obj.population("default")
        assert pop is pop2


class TestEdgePopulation(object):

    @staticmethod
    def mocking_nodes(pop_name):
        node_population = Mock()
        node_population.name = pop_name
        node_population.ids = lambda x: x
        return node_population

    @staticmethod
    def create_population(filepath, pop_name):
        config = {
            'edges_file': filepath,
            'edge_types_file': None,
        }
        node_population = TestEdgePopulation.mocking_nodes("default")
        circuit = Mock()
        circuit.nodes = {node_population.name: node_population}

        storage = test_module.EdgeStorage(config, circuit)
        return storage.population(pop_name)

    def setup(self):
        self.test_obj = TestEdgePopulation.create_population(
            os.path.join(TEST_DATA_DIR, "edges.h5"), 'default')

    def test_basic(self):
        assert self.test_obj.h5_filepath == os.path.join(TEST_DATA_DIR, 'edges.h5')
        assert self.test_obj.name == 'default'
        assert self.test_obj.source_name == 'default'
        assert self.test_obj.target_name == 'default'
        assert self.test_obj.size, 4
        assert (
                sorted(self.test_obj.property_names) ==
                sorted([
                    Synapse.AXONAL_DELAY,
                    Synapse.G_SYNX,
                    Synapse.POST_X_CENTER,
                    Synapse.POST_Y_CENTER,
                    Synapse.POST_Z_CENTER,
                    Synapse.POST_X_SURFACE,
                    Synapse.POST_Y_SURFACE,
                    Synapse.POST_Z_SURFACE,
                    Synapse.PRE_X_CENTER,
                    Synapse.PRE_Y_CENTER,
                    Synapse.PRE_Z_CENTER,
                    Synapse.PRE_X_SURFACE,
                    Synapse.PRE_Y_SURFACE,
                    Synapse.PRE_Z_SURFACE,
                    test_module.DYNAMICS_PREFIX + 'param1'
                ])
        )

    def test_container_properties(self):
        expected = sorted(
            ['PRE_Y_SURFACE', 'PRE_Z_SURFACE', 'PRE_X_CENTER', 'POST_Y_CENTER', 'AXONAL_DELAY',
             'POST_X_CENTER', 'POST_Y_SURFACE', 'POST_Z_SURFACE', 'PRE_Y_CENTER', 'POST_Z_CENTER',
             'PRE_Z_CENTER', 'PRE_X_SURFACE', 'POST_X_SURFACE'])
        assert sorted(self.test_obj.container_property_names(Edge)) == expected
        with pytest.raises(BluepySnapError):
            mapping = {"X": "x"}
            self.test_obj.container_property_names(mapping)

        with pytest.raises(BluepySnapError):
            self.test_obj.container_property_names(int)

    def test_nodes_1(self):
        assert self.test_obj._nodes('default').name == 'default'
        with pytest.raises(BluepySnapError):
            self.test_obj._nodes('no-such-population')

    def test_nodes_2(self):
        self.test_obj._edge_storage.circuit.nodes = {
            'A': 'aa',
            'B': 'bb',
        }
        assert self.test_obj._nodes('B') == 'bb'
        with pytest.raises(BluepySnapError):
            self.test_obj._nodes('no-such-population')

    def test_properties_1(self):
        properties = [
            Synapse.PRE_GID,
            Synapse.POST_GID,
            Synapse.AXONAL_DELAY,
            Synapse.POST_X_CENTER,
            test_module.DYNAMICS_PREFIX + 'param1',
        ]
        edge_ids = [0, 1]
        actual = self.test_obj.properties(edge_ids, properties)
        expected = pd.DataFrame(
            [
                (2, 0, 99.8945, 1110., 0.),
                (0, 1, 88.1862, 1111., 1.),
            ],
            columns=properties,
            index=index_as_uint64(edge_ids)
        )
        pdt.assert_frame_equal(actual, expected, check_dtype=False)

    def test_properties_2(self):
        prop = Synapse.AXONAL_DELAY
        edge_ids = [1, 0]
        actual = self.test_obj.properties(edge_ids, prop)
        expected = pd.Series([88.1862, 99.8945], index=index_as_uint64(edge_ids), name=prop)
        pdt.assert_series_equal(actual, expected, check_dtype=False)

    def test_properties_3(self):
        properties = [Synapse.PRE_GID, Synapse.AXONAL_DELAY]
        pdt.assert_series_equal(
            self.test_obj.properties([], properties[0]),
            pd.Series(name=properties[0])
        )
        pdt.assert_frame_equal(
            self.test_obj.properties([], properties),
            pd.DataFrame(columns=properties)
        )

    def test_properties_4(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.properties([0], 'no-such-property')

    def test_positions_1(self):
        actual = self.test_obj.positions([0], 'afferent', 'center')
        expected = pd.DataFrame([
            [1110., 1120., 1130.]
        ],
            index=index_as_uint64([0]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_2(self):
        actual = self.test_obj.positions([1], 'afferent', 'surface')
        expected = pd.DataFrame([
            [1211., 1221., 1231.]
        ],
            index=index_as_uint64([1]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_3(self):
        actual = self.test_obj.positions([2], 'efferent', 'center')
        expected = pd.DataFrame([
            [2112., 2122., 2132.]
        ],
            index=index_as_uint64([2]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_4(self):
        actual = self.test_obj.positions([3], 'efferent', 'surface')
        expected = pd.DataFrame([
            [2213., 2223., 2233.]
        ],
            index=index_as_uint64([3]),
            columns=['x', 'y', 'z']
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_5(self):
        with pytest.raises(AssertionError):
            self.test_obj.positions([2], 'err', 'center')

    def test_positions_6(self):
        with pytest.raises(AssertionError):
            self.test_obj.positions([2], 'afferent', 'err')

    def test_afferent_nodes(self):
        npt.assert_equal(self.test_obj.afferent_nodes(0), [2])
        npt.assert_equal(self.test_obj.afferent_nodes(1), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes(2), [])

    def test_efferent_nodes(self):
        npt.assert_equal(self.test_obj.efferent_nodes(0), [1])
        npt.assert_equal(self.test_obj.efferent_nodes(1), [])
        npt.assert_equal(self.test_obj.efferent_nodes(2), [0, 1])

    def test_afferent_edges_1(self):
        npt.assert_equal(
            self.test_obj.afferent_edges(1, None),
            [1, 2, 3]
        )

    def test_afferent_edges_2(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.afferent_edges(1, properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                    [11.1058],
                ],
                columns=properties, index=index_as_uint64([1, 2, 3])
            ),
            check_dtype=False
        )

    def test_efferent_edges_1(self):
        npt.assert_equal(
            self.test_obj.efferent_edges(2, None),
            [0, 3]
        )

    def test_efferent_edges_2(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.efferent_edges(2, properties),
            pd.DataFrame(
                [
                    [99.8945],
                    [11.1058],
                ],
                columns=properties, index=index_as_uint64([0, 3])
            ),
            check_dtype=False
        )

    def test_pair_edges_1(self):
        npt.assert_equal(self.test_obj.pair_edges(0, 2, None), [])

    def test_pair_edges_2(self):
        actual = self.test_obj.pair_edges(0, 2, [Synapse.AXONAL_DELAY])
        assert actual.empty

    def test_pair_edges_3(self):
        npt.assert_equal(self.test_obj.pair_edges(2, 0, None), [0])

    def test_pair_edges_4(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.pair_edges(2, 0, properties),
            pd.DataFrame(
                [
                    [99.8945],
                ],
                columns=properties, index=index_as_uint64([0])
            ),
            check_dtype=False
        )

    def test_pathway_edges_1(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.pathway_edges([0, 1], [1, 2], properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                ],
                columns=properties, index=index_as_uint64([1, 2])
            ),
            check_dtype=False
        )

    def test_pathway_edges_2(self):
        npt.assert_equal(
            self.test_obj.pathway_edges([1, 2], [0, 2], None),
            [0]
        )

    def test_pathway_edges_3(self):
        npt.assert_equal(
            self.test_obj.pathway_edges([0, 1], None, None),
            [1, 2]
        )

    def test_pathway_edges_4(self):
        npt.assert_equal(
            self.test_obj.pathway_edges(None, [0, 1], None),
            [0, 1, 2, 3]
        )

    def test_pathway_edges_5(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.pathway_edges(None, None, None)

    def test_iter_connections_1(self):
        it = self.test_obj.iter_connections(
            [0, 2], [1]
        )
        assert next(it) == (0, 1)
        assert next(it) == (2, 1)
        with pytest.raises(StopIteration):
            next(it)

    def test_iter_connections_2(self):
        it = self.test_obj.iter_connections(
            [0, 2], [1], unique_node_ids=True
        )
        assert list(it) == [(0, 1)]

    def test_iter_connections_3(self):
        it = self.test_obj.iter_connections(
            [0, 2], [1], shuffle=True
        )
        assert sorted(it) == [(0, 1), (2, 1)]

    def test_iter_connections_4(self):
        it = self.test_obj.iter_connections(
            None, None
        )
        with pytest.raises(BluepySnapError):
            next(it)

    def test_iter_connections_5(self):
        it = self.test_obj.iter_connections(
            None, [1]
        )
        assert list(it) == [(0, 1), (2, 1)]

    def test_iter_connections_6(self):
        it = self.test_obj.iter_connections(
            [2], None
        )
        assert list(it) == [(2, 0), (2, 1)]

    def test_iter_connections_7(self):
        it = self.test_obj.iter_connections(
            [], [0, 1, 2]
        )
        assert list(it) == []

    def test_iter_connections_8(self):
        it = self.test_obj.iter_connections(
            [0, 2], [1], return_edge_ids=True
        )
        npt.assert_equal(list(it), [(0, 1, [1, 2]), (2, 1, [3])])

    def test_iter_connections_9(self):
        it = self.test_obj.iter_connections(
            [0, 2], [1], return_edge_count=True
        )
        assert list(it) == [(0, 1, 2), (2, 1, 1)]

    def test_iter_connections_10(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.iter_connections(
                [0, 2], [1], return_edge_ids=True, return_edge_count=True
            )
