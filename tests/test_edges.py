import mock
import numpy as np
import numpy.testing as npt
from numpy import dtype
import pandas as pd
import pandas.testing as pdt
import pytest

import libsonata
from mock import Mock, patch, PropertyMock

from bluepysnap.bbp import Synapse
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Edge
from bluepysnap.node_sets import NodeSets
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids import CircuitEdgeId, CircuitEdgeIds

import bluepysnap.edges as test_module

from utils import TEST_DATA_DIR, create_node_population


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


class TestEdges:
    def setup(self):
        circuit = Circuit(str(TEST_DATA_DIR / 'circuit_config.json'))
        self.test_obj = test_module.Edges(circuit)

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.EdgePopulation)
        with pytest.raises(BluepySnapError):
            self.test_obj["unknown"]

    def test_iter(self):
        assert sorted(self.test_obj) == ['default', 'default2']

    def test_population_names(self):
        assert self.test_obj.population_names == ['default', 'default2']

    def test_keys_names(self):
        assert list(self.test_obj.keys()) == ['default', 'default2']
        assert list(self.test_obj.names()) == list(self.test_obj.keys())

    def test_values_population(self):
        values = list(self.test_obj.values())
        assert isinstance(values[0], test_module.EdgePopulation)
        assert values[0].name == 'default'

        assert isinstance(values[1], test_module.EdgePopulation)
        assert values[1].name == 'default2'

        assert list(self.test_obj.values()) == list(self.test_obj.populations())

    def test_items(self):
        keys, values = zip(*self.test_obj.items())
        assert keys == ('default', 'default2')
        assert isinstance(values[0], test_module.EdgePopulation)
        assert values[0].name == 'default'

        assert isinstance(values[1], test_module.EdgePopulation)
        assert values[1].name == 'default2'

    def test_size(self):
        assert self.test_obj.size == 8

    def test_property_names(self):
        assert self.test_obj.property_names == {'@dynamics:param1', '@source_node', '@target_node',
                                                'afferent_center_x', 'afferent_center_y',
                                                'afferent_center_z', 'afferent_section_id',
                                                'afferent_section_pos', 'afferent_surface_x',
                                                'afferent_surface_y', 'afferent_surface_z',
                                                'conductance', 'delay', 'efferent_center_x',
                                                'efferent_center_y', 'efferent_center_z',
                                                'efferent_section_id', 'efferent_section_pos',
                                                'efferent_surface_x', 'efferent_surface_y',
                                                'efferent_surface_z', 'other1', 'other2',
                                                'syn_weight'}

    def test_property_dtypes(self):
        expected = pd.Series(
            data=[dtype('float32'), dtype('float64'), dtype('float64'), dtype('float64'), dtype(
                'float32'), dtype('float64'), dtype('float32'), dtype('float64'), dtype(
                'int64'), dtype('int64'), dtype('float64'), dtype('float64'), dtype(
                'float64'), dtype('float64'), dtype('float64'), dtype('float64'), dtype(
                'float32'), dtype('float32'), dtype('float64'), dtype('float64'),
                dtype('uint64'), dtype('uint64'), dtype('O'), dtype('int32')]
            , index=['syn_weight', '@dynamics:param1', 'afferent_surface_y',
                     'afferent_surface_z', 'conductance', 'efferent_center_x',
                     'delay', 'afferent_center_z', 'efferent_section_id',
                     'afferent_section_id', 'efferent_center_y',
                     'afferent_center_x', 'efferent_surface_z',
                     'afferent_center_y', 'afferent_surface_x',
                     'efferent_surface_x', 'afferent_section_pos',
                     'efferent_section_pos', 'efferent_surface_y',
                     'efferent_center_z',
                     '@source_node', '@target_node', 'other1', 'other2']).sort_index()
        pdt.assert_series_equal(self.test_obj.property_dtypes.sort_index(), expected)

    def test_property_dtypes_fail(self):
        a = pd.Series(data=[dtype('int64'), dtype('float64')],
                      index=['syn_weight', 'efferent_surface_z']).sort_index()
        b = pd.Series(data=[dtype('int32'), dtype('float64')],
                      index=['syn_weight', 'efferent_surface_z']).sort_index()

        with patch("bluepysnap.edges.EdgePopulation.property_dtypes", new_callable=PropertyMock) as mock:
            mock.side_effect = [a, b]
            circuit = Circuit(str(TEST_DATA_DIR / 'circuit_config.json'))
            test_obj = test_module.Edges(circuit)
            with pytest.raises(BluepySnapError):
                test_obj.property_dtypes.sort_index()

    def test_ids(self):

        # single edge ID --> CircuitEdgeIds return populations with the 0 id
        expected = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)])
        assert self.test_obj.ids(0) ==  expected

        # single edge ID list --> CircuitEdgeIds return populations with the 0 id
        expected = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)])
        assert self.test_obj.ids([0]) == expected

        # default3 population does not exist and is asked explicitly
        with pytest.raises(BluepySnapError):
            ids = CircuitEdgeIds.from_arrays(["default", "default3"], [0, 3])
            self.test_obj.ids(ids)

        # seq of node ID --> CircuitNodeIds return populations with the array of ids
        expected = CircuitEdgeIds.from_arrays(["default", "default", "default2", "default2"],
                                              [0, 1, 0, 1])
        tested = self.test_obj.ids([0, 1])
        assert tested == expected
        tested = self.test_obj.ids((0, 1))
        assert tested == expected
        tested = self.test_obj.ids(np.array([0, 1]))
        assert tested == expected

        # Check operations on global ids
        ids = self.test_obj.ids([0, 1, 2, 3])
        assert ids.filter_population("default").append(ids.filter_population("default2")) == ids

        expected = CircuitEdgeIds.from_arrays(["default2", "default2"], [0, 1])
        assert ids.filter_population("default2").limit(2) == expected

    def test_properties(self):
        ids = CircuitEdgeIds.from_dict({"default": [0,1,2,3], "default2": [0,1,2,3]})
        tested = self.test_obj.properties(ids, properties=self.test_obj.property_names)
        assert len(tested) == 8
        assert len(list(tested)) == 24

        # put NaN for the undefined values : only values for default2 in dropna
        assert len(tested.dropna()) == 4

        # the index of the dataframe is indentical to the CircuitEdgeIds index
        pdt.assert_index_equal(tested.index, ids.index)
        pdt.assert_frame_equal(self.test_obj.properties([0,1,2,3], properties=self.test_obj.property_names), tested)

        # tested columns
        tested = self.test_obj.properties(ids, properties=["other2", "other1", "@source_node"])
        assert tested.shape == (self.test_obj.size, 3)
        assert list(tested) == ["other2", "other1", "@source_node"]

        tested = self.test_obj.properties(CircuitEdgeIds.from_dict({"default2": [0,1,2,3]}),
                                   properties=["other2", "other1", "@source_node"])
        assert tested.shape == (4, 3)
        # correct ordering when setting the dataframe with the population dataframe
        assert tested.loc[("default2", 0)].tolist() == [10, 'A', 2]
        with pytest.raises(KeyError):
            tested.loc[("default", 0)]

        tested = self.test_obj.properties(CircuitEdgeIds.from_dict({"default": [0,1,2,3]}),
                                   properties=["other2", "other1", '@source_node'])
        assert tested.shape == (4, 3)
        assert tested.loc[("default", 0)].tolist() == [np.NaN, np.NaN, 2]
        assert tested.loc[("default", 1)].tolist() == [np.NaN, np.NaN, 0]

        tested = self.test_obj.properties(ids, properties='@source_node')
        assert tested["@source_node"].tolist() == [2, 0, 0, 2, 2, 0, 0, 2]

        tested = self.test_obj.properties(ids, properties='other2')
        assert tested["other2"].tolist() == [np.NaN, np.NaN, np.NaN, np.NaN, 10, 11, 12, 13]

        with pytest.raises(BluepySnapError):
            self.test_obj.properties(ids, properties=["other2", "unknown"])

        with pytest.raises(BluepySnapError):
            self.test_obj.properties(ids, properties="unknown")

    def test_afferent_nodes(self):
        assert self.test_obj.afferent_nodes(0) == CircuitEdgeIds.from_arrays(["default", "default2"], [2, 2])
        assert self.test_obj.afferent_nodes([0, 1]) == CircuitEdgeIds.from_dict({"default": [2, 0], "default2": [2,0]})
        ids = CircuitNodeIds.from_dict({"default": [0, 1], "default2": [0, 1]})
        assert self.test_obj.afferent_nodes(ids) == CircuitEdgeIds.from_dict({"default": [2, 0], "default2": [2,0]})




class TestEdgeStorage:
    def setup(self):
        config = {
            'edges_file': str(TEST_DATA_DIR / 'edges.h5'),
            'edge_types_file': None,
        }
        self.circuit = Mock()
        self.test_obj = test_module.EdgeStorage(config, self.circuit)

    def test_storage(self):
        assert isinstance(self.test_obj.storage, libsonata.EdgeStorage)

    def test_population_names(self):
        assert sorted(list(self.test_obj.population_names)) == ["default", "default2"]

    def test_circuit(self):
        assert self.test_obj.circuit is self.circuit

    def test_population(self):
        pop = self.test_obj.population("default")
        assert isinstance(pop, test_module.EdgePopulation)
        assert pop.name == "default"
        pop2 = self.test_obj.population("default")
        assert pop is pop2


class TestEdgePopulation:

    @staticmethod
    def create_edge_population(filepath, pop_name):
        config = {
            'edges_file': filepath,
            'edge_types_file': None,
        }
        circuit = Mock()
        create_node_population(str(TEST_DATA_DIR / 'nodes.h5'), "default", circuit=circuit,
                               node_sets=NodeSets(str(TEST_DATA_DIR / 'node_sets.json')))
        storage = test_module.EdgeStorage(config, circuit)
        pop = storage.population(pop_name)

        # check if the source and target populations are in the circuit nodes
        assert pop.source.name in pop._edge_storage.circuit.nodes
        assert pop.target.name in pop._edge_storage.circuit.nodes
        return pop

    def setup(self):
        self.test_obj = TestEdgePopulation.create_edge_population(
            str(TEST_DATA_DIR / "edges.h5"), 'default')

    def test_basic(self):
        assert self.test_obj._edge_storage._h5_filepath == str(TEST_DATA_DIR / 'edges.h5')
        assert self.test_obj.name == 'default'
        assert self.test_obj.source.name == 'default'
        assert self.test_obj.target.name == 'default'
        assert self.test_obj.size, 4
        assert (
                sorted(self.test_obj.property_names) ==
                sorted([
                    Synapse.SOURCE_NODE_ID,
                    Synapse.TARGET_NODE_ID,
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
                    Synapse.POST_SECTION_ID,
                    Synapse.POST_SECTION_POS,
                    Synapse.PRE_SECTION_ID,
                    Synapse.PRE_SECTION_POS,
                    Synapse.SYN_WEIGHT,
                    test_module.DYNAMICS_PREFIX + 'param1'
                ])
        )

    def test_container_properties(self):
        expected = sorted(
            ['PRE_Y_SURFACE', 'PRE_Z_SURFACE', 'PRE_X_CENTER', 'POST_Y_CENTER', 'AXONAL_DELAY',
             'POST_X_CENTER', 'POST_Y_SURFACE', 'POST_Z_SURFACE', 'PRE_Y_CENTER', 'POST_Z_CENTER',
             'PRE_Z_CENTER', 'PRE_X_SURFACE', 'POST_X_SURFACE', 'POST_SECTION_ID', 'PRE_SECTION_ID',
             'POST_SECTION_POS', 'PRE_SECTION_POS', 'SYN_WEIGHT',
             'SOURCE_NODE_ID', 'TARGET_NODE_ID'])
        assert sorted(self.test_obj.container_property_names(Edge)) == expected
        with pytest.raises(BluepySnapError):
            mapping = {"X": "x"}
            self.test_obj.container_property_names(mapping)

        with pytest.raises(BluepySnapError):
            self.test_obj.container_property_names(int)

    def test_nodes(self):
        assert self.test_obj._nodes('default').name == 'default'
        with pytest.raises(BluepySnapError):
            self.test_obj._nodes('no-such-population')

    def test_property_dtypes(self):
        from numpy import dtype
        expected = pd.Series(
            data=[dtype('float32'), dtype('float64'), dtype('float64'), dtype('float64'), dtype(
                'float32'), dtype('float64'), dtype('float32'), dtype('float64'), dtype(
                'int64'), dtype('int64'), dtype('float64'), dtype('float64'), dtype(
                'float64'), dtype('float64'), dtype('float64'), dtype('float64'), dtype(
                'float32'), dtype('float32'), dtype('float64'), dtype('float64'),
                dtype('uint64'), dtype('uint64')]
            , index=['syn_weight', '@dynamics:param1', 'afferent_surface_y',
                     'afferent_surface_z', 'conductance', 'efferent_center_x',
                     'delay', 'afferent_center_z', 'efferent_section_id',
                     'afferent_section_id', 'efferent_center_y',
                     'afferent_center_x', 'efferent_surface_z',
                     'afferent_center_y', 'afferent_surface_x',
                     'efferent_surface_x', 'afferent_section_pos',
                     'efferent_section_pos', 'efferent_surface_y',
                     'efferent_center_z',
                     '@source_node', '@target_node']).sort_index()

        pdt.assert_series_equal(expected, self.test_obj.property_dtypes)

    def test_ids(self):
        assert self.test_obj.ids(0) == [0]
        npt.assert_equal(self.test_obj.ids([0, 1]), np.array([0, 1]))
        npt.assert_equal(self.test_obj.ids(np.array([0, 1])), np.array([0, 1]))
        npt.assert_equal(self.test_obj.ids(CircuitEdgeId("default", 0)), [0])
        npt.assert_equal(self.test_obj.ids(CircuitEdgeId("default2", 0)), [])
        ids = CircuitEdgeIds.from_tuples([("default", 0), ("default", 1)])
        npt.assert_equal(self.test_obj.ids(ids), np.array([0, 1]))
        ids = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 1)])
        npt.assert_equal(self.test_obj.ids(ids), np.array([0]))
        ids = CircuitEdgeIds.from_tuples([("default2", 0), ("default2", 1)])
        npt.assert_equal(self.test_obj.ids(ids), [])


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
            pd.Series(name=properties[0], dtype=np.float64)
        )
        pdt.assert_frame_equal(
            self.test_obj.properties([], properties),
            pd.DataFrame(columns=properties)
        )

    def test_properties_4(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.properties([0], 'no-such-property')

    def test_properties_all_edge_ids_types(self):

        assert self.test_obj.properties(0, Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.properties([0], Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.properties([0, 1], Synapse.PRE_GID).tolist() == [2, 0]
        assert self.test_obj.properties(CircuitEdgeId("default", 0), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.properties(CircuitEdgeIds.from_tuples([("default", 0)]), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.properties(CircuitEdgeId("default2", 0), Synapse.PRE_GID).tolist() == []
        assert self.test_obj.properties(CircuitEdgeIds.from_tuples([("default", 0), ("default", 1)]), Synapse.PRE_GID).tolist() == [2, 0]
        assert self.test_obj.properties(CircuitEdgeIds.from_tuples([("default", 0), ("default2", 1)]), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.properties(CircuitEdgeIds.from_tuples([("default2", 0), ("default2", 1)]), Synapse.PRE_GID).tolist() == []

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
        npt.assert_equal(self.test_obj.afferent_nodes(0, unique=False), [2])

        npt.assert_equal(self.test_obj.afferent_nodes(1), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes(1, unique=False), [0, 0, 2])

        npt.assert_equal(self.test_obj.afferent_nodes(2), [])

        npt.assert_equal(self.test_obj.afferent_nodes([0, 1]), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes([0, 1], unique=False), [2, 0, 0, 2])

        npt.assert_equal(self.test_obj.afferent_nodes({}), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes({'mtype': 'L2_X'}), [2])  # eq node id 0 as target
        npt.assert_equal(self.test_obj.afferent_nodes({'mtype': 'L2_X'}), [2])  # eq node id 0 as target

        npt.assert_equal(self.test_obj.afferent_nodes(None), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes(None, unique=False), [2, 0, 0, 2])

    def test_efferent_nodes(self):
        npt.assert_equal(self.test_obj.efferent_nodes(0), [1])
        npt.assert_equal(self.test_obj.efferent_nodes(0, unique=False), [1, 1])

        npt.assert_equal(self.test_obj.efferent_nodes(1), [])
        npt.assert_equal(self.test_obj.efferent_nodes(1, unique=False), [])

        npt.assert_equal(self.test_obj.efferent_nodes(2), [0, 1])
        npt.assert_equal(self.test_obj.efferent_nodes(2, unique=False), [0, 1])

        npt.assert_equal(self.test_obj.efferent_nodes([0, 1]), [1])
        npt.assert_equal(self.test_obj.efferent_nodes([0, 1], unique=False), [1, 1])

        npt.assert_equal(self.test_obj.efferent_nodes({}), [0, 1])
        npt.assert_equal(self.test_obj.efferent_nodes({'mtype': 'L2_X'}), [1])  # eq node id 0 as source
        npt.assert_equal(self.test_obj.efferent_nodes({'mtype': 'L2_X'}), [1])  # eq node id 0 as source

        npt.assert_equal(self.test_obj.efferent_nodes(None), [0, 1])
        npt.assert_equal(self.test_obj.efferent_nodes(None, unique=False), [0, 1, 1, 1])

    def test_afferent_edges(self):
        npt.assert_equal(
            self.test_obj.afferent_edges([0, 1], None),
            [0, 1, 2, 3]
        )

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

    def test_pathway_edges_6(self):
        ids = CircuitNodeIds.from_dict({"default": [0, 1]})
        npt.assert_equal(self.test_obj.pathway_edges(ids, None, None), [1, 2])

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

    def test_iter_connection_unique(self):
        test_obj = TestEdgePopulation.create_edge_population(
            str(TEST_DATA_DIR / "edges_complete_graph.h5"), 'default')
        it = test_obj.iter_connections([0, 1, 2], [0, 1, 2])
        assert sorted(it) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        it = test_obj.iter_connections([0, 1, 2], [0, 1, 2], unique_node_ids=True)
        assert sorted(it) == [(0, 1), (1, 0)]

        it = test_obj.iter_connections([0, 1, 2], [0, 2], unique_node_ids=True)
        assert sorted(it) == [(0, 2), (1, 0)]

        it = test_obj.iter_connections([0, 2], [0, 2], unique_node_ids=True)
        assert sorted(it) == [(0, 2), (2, 0)]

        it = test_obj.iter_connections([0, 1, 2], [0, 2, 1], unique_node_ids=True)
        assert sorted(it) == [(0, 1), (1, 0)]

        it = test_obj.iter_connections([1, 2], [0, 1, 2], unique_node_ids=True)
        assert sorted(it) == [(1, 0), (2, 1)]

        it = test_obj.iter_connections([0, 1, 2], [1, 2], unique_node_ids=True)
        assert sorted(it) == [(0, 1), (1, 2)]
