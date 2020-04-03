import json

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
from pandas.api.types import is_categorical

import pytest
import libsonata
from mock import Mock

from bluepysnap.bbp import Cell
from bluepysnap.sonata_constants import Node
from bluepysnap.node_sets import NodeSets
from bluepysnap.exceptions import BluepySnapError

import bluepysnap.nodes as test_module

from utils import TEST_DATA_DIR


class TestNodeStorage:
    def setup(self):
        config = {
            'nodes_file': str(TEST_DATA_DIR / 'nodes.h5'),
            'node_types_file': None
        }
        self.circuit = Mock()
        self.test_obj = test_module.NodeStorage(config, self.circuit)

    def test_storage(self):
        assert isinstance(self.test_obj.storage, libsonata.NodeStorage)

    def test_population_names(self):
        assert sorted(list(self.test_obj.population_names)) == ["default", "default2"]

    def test_circuit(self):
        assert self.test_obj.circuit is self.circuit

    def test_population(self):
        pop = self.test_obj.population("default")
        assert isinstance(pop, test_module.NodePopulation)
        assert pop.name == "default"
        pop2 = self.test_obj.population("default")
        assert pop is pop2

    def test_load_population_data(self):
        data = self.test_obj.load_population_data("default")
        assert isinstance(data, pd.DataFrame)
        assert sorted(list(data)) == sorted(['layer', 'morphology', 'mtype', 'rotation_angle_xaxis',
                                             'rotation_angle_yaxis', 'rotation_angle_zaxis', 'x',
                                             'y', 'z', 'model_template', 'model_type',
                                             '@dynamics:holding_current'])
        assert len(data) == 3


class TestNodePopulation:

    @staticmethod
    def create_population(filepath, pop_name):
        config = {
            'nodes_file': filepath,
            'node_types_file': None,
        }
        circuit = Mock()
        circuit.node_sets = NodeSets(str(TEST_DATA_DIR / 'node_sets.json'))
        storage = test_module.NodeStorage(config, circuit)
        return storage.population(pop_name)

    def setup(self):
        self.test_obj = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes.h5'),
            "default")

    def test_basic(self):
        assert self.test_obj._node_storage._h5_filepath == str(TEST_DATA_DIR / 'nodes.h5')
        assert self.test_obj.name == 'default'
        assert self.test_obj.size == 3
        assert (
                sorted(self.test_obj.property_names) ==
                [
                    Cell.HOLDING_CURRENT,
                    Cell.LAYER,
                    Cell.MODEL_TEMPLATE,
                    Cell.MODEL_TYPE,
                    Cell.MORPHOLOGY,
                    Cell.MTYPE,
                    Cell.ROTATION_ANGLE_X,
                    Cell.ROTATION_ANGLE_Y,
                    Cell.ROTATION_ANGLE_Z,
                    Cell.X,
                    Cell.Y,
                    Cell.Z,
                ]
        )
        assert (sorted(self.test_obj._node_sets) ==
                sorted(json.load(open(str(TEST_DATA_DIR / 'node_sets.json')))))

    def test_property_values(self):
        assert (
                self.test_obj.property_values(Cell.LAYER) ==
                {2, 6}
        )
        assert (
                self.test_obj.property_values(Cell.MORPHOLOGY) ==
                {'morph-A', 'morph-B', 'morph-C'}
        )
        test_obj_library = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_with_library.h5'),
            "default")
        assert test_obj_library.property_values("categorical") == {"A", "B", "C"}
        assert test_obj_library.property_values("categorical", is_present=True) == {"A", "B"}

    def test_property_dtypes(self):
        from numpy import dtype

        expected = pd.Series(data=[dtype('int64'), dtype('O'), dtype('O'), dtype('O'), dtype('O'),
                                   dtype('float64'), dtype('float64'), dtype('float64'),
                                   dtype('float64'),
                                   dtype('float64'), dtype('float64'), dtype('float64')],
                             index=['layer', 'model_template', 'model_type', 'morphology', 'mtype',
                                    'rotation_angle_xaxis', 'rotation_angle_yaxis',
                                    'rotation_angle_zaxis',
                                    'x', 'y', 'z', '@dynamics:holding_current']).sort_index()

        pdt.assert_series_equal(expected, self.test_obj.property_dtypes)

    def test_container_properties(self):
        expected = sorted(['X', 'Y', 'Z', 'MORPHOLOGY', 'HOLDING_CURRENT', 'ROTATION_ANGLE_X',
                           'ROTATION_ANGLE_Y', 'ROTATION_ANGLE_Z', 'MTYPE', 'LAYER',
                           'MODEL_TEMPLATE', 'MODEL_TYPE'])
        assert sorted(self.test_obj.container_property_names(Cell)) == expected
        expected = sorted(['X', 'Y', 'Z', 'MORPHOLOGY', 'ROTATION_ANGLE_X', 'ROTATION_ANGLE_Y',
                           'ROTATION_ANGLE_Z', 'MODEL_TEMPLATE', 'MODEL_TYPE'])
        assert sorted(self.test_obj.container_property_names(Node)) == expected

        with pytest.raises(BluepySnapError):
            mapping = {"X": "x"}
            self.test_obj.container_property_names(mapping)

        with pytest.raises(BluepySnapError):
            self.test_obj.container_property_names(int)

    def test__positional_mask(self):
        npt.assert_array_equal(self.test_obj._positional_mask([1, 2]), [False, True, True])
        npt.assert_array_equal(self.test_obj._positional_mask([0, 2]), [True, False, True])

    def test__node_population_mask(self):
        queries, mask = self.test_obj._node_population_mask({"population": "default",
                                                             "other": "val"})
        assert queries == {"other": "val"}
        npt.assert_array_equal(mask, [True, True, True])

        queries, mask = self.test_obj._node_population_mask({"population": "unknown",
                                                             "other": "val"})
        assert queries == {"other": "val"}
        npt.assert_array_equal(mask, [False, False, False])

        queries, mask = self.test_obj._node_population_mask({"population": "default",
                                                             "node_id": [2], "other": "val"})
        assert queries == {"other": "val"}
        npt.assert_array_equal(mask, [False, False, True])

        queries, mask = self.test_obj._node_population_mask({"other": "val"})
        assert queries == {"other": "val"}
        npt.assert_array_equal(mask, [True, True, True])

    def test_ids(self):
        _call = self.test_obj.ids
        npt.assert_equal(_call(), [0, 1, 2])
        npt.assert_equal(_call(group={}), [0, 1, 2])
        npt.assert_equal(_call(group=[]), [])
        npt.assert_equal(_call(limit=1), [0])
        npt.assert_equal(len(_call(sample=2)), 2)
        npt.assert_equal(_call(0), [0])
        npt.assert_equal(_call([0, 1]), [0, 1])
        npt.assert_equal(_call([1, 0, 1]), [1, 0, 1])  # order and duplicates preserved
        npt.assert_equal(_call(np.array([1, 0, 1])), np.array([1, 0, 1]))

        npt.assert_equal(_call({Cell.MTYPE: 'L6_Y'}), [1, 2])
        npt.assert_equal(_call({Cell.X: (100, 203)}), [0, 1])
        npt.assert_equal(_call({Cell.MTYPE: 'L6_Y', Cell.MORPHOLOGY: "morph-B"}), [1])
        # same query with a $and operator
        npt.assert_equal(_call({"$and": [{Cell.MTYPE: 'L6_Y'}, {Cell.MORPHOLOGY: "morph-B"}]}), [1])
        npt.assert_equal(_call({Cell.MORPHOLOGY: ['morph-A', 'morph-B']}), [0, 1])
        npt.assert_equal(_call({"$and": [{}, {}]}), [0, 1, 2])
        npt.assert_equal(_call({"$and": [{}, {Cell.MORPHOLOGY: 'morph-B'}]}), [1])
        # same query with a $or operator
        npt.assert_equal(_call({"$or": [{Cell.MORPHOLOGY: 'morph-A'},
                                        {Cell.MORPHOLOGY: 'morph-B'}]}), [0, 1])
        npt.assert_equal(_call({"$or": [{Cell.MTYPE: 'L6_Y'},
                                        {Cell.MORPHOLOGY: "morph-B"}]}), [1, 2])
        npt.assert_equal(_call({"$or": [{}, {}]}), [0, 1, 2])
        npt.assert_equal(_call({"$or": [{}, {Cell.MORPHOLOGY: 'morph-B'}]}), [0, 1, 2])
        # non destructive operation for queries
        query = {"$and": [{"$or": [{Cell.MTYPE: 'L6_Y'}, {Cell.MORPHOLOGY: "morph-B"}]},
                          {"node_id": [1]}]}
        npt.assert_equal(_call(query), [1])
        npt.assert_equal(_call(query), [1])

        npt.assert_equal(_call('Layer2'), [0])
        npt.assert_equal(_call('Layer23'), [0])
        npt.assert_equal(_call('Empty_nodes'), [])
        npt.assert_equal(_call('Node2012'), [0, 1, 2])  # reordered + duplicates are removed
        npt.assert_equal(_call('Node12_L6_Y'), [1, 2])
        npt.assert_equal(_call('Node2_L6_Y'), [2])

        npt.assert_equal(_call('Node0_L6_Y'), [])  # return empty if disjoint samples
        npt.assert_equal(_call('Empty_L6_Y'), [])  # return empty if empty node_id = []
        npt.assert_equal(_call('Population_default'), [0, 1, 2])  # return all ids
        npt.assert_equal(_call('Population_default2'), [])  # return empty if diff population
        npt.assert_equal(_call('Population_default_L6_Y'), [1, 2])  # population + other query ok
        # population + other query + node_id ok
        npt.assert_equal(_call('Population_default_L6_Y_Node2'), [2])
        npt.assert_equal(_call('combined_Node0_L6_Y__Node12_L6_Y'), [1, 2])  # 'or' function
        npt.assert_equal(_call('combined_combined_Node0_L6_Y__Node12_L6_Y__'),
                         [0, 1, 2])  # imbricated '$or' functions

        with pytest.raises(BluepySnapError):
            _call('no-such-node-set')
        with pytest.raises(BluepySnapError):
            _call(-1)  # node ID out of range (lower boundary)
        with pytest.raises(BluepySnapError):
            _call(999)  # node ID out of range (upper boundary)
        with pytest.raises(BluepySnapError):
            _call([1, 999])  # one of node IDs out of range
        with pytest.raises(BluepySnapError):
            _call({'no-such-node-property': 42})

    def test_node_ids_by_filter_complex_query(self):
        test_obj = TestNodePopulation.create_population(str(TEST_DATA_DIR / 'nodes.h5'), "default")
        data = pd.DataFrame({
            Cell.MTYPE: ['L23_MC', 'L4_BP', 'L6_BP', 'L6_BPC'],
        })
        # replace the data using the __dict__ directly
        test_obj.__dict__["_data"] = data

        # only full match is accepted
        npt.assert_equal(
            [1, 2],
            test_obj.ids({Cell.MTYPE: {'$regex': '.*BP'}, })
        )
        # ...not 'startswith'
        npt.assert_equal(
            [],
            test_obj.ids({
                Cell.MTYPE: {'$regex': 'L6'}, })
        )
        # ...or 'endswith'
        npt.assert_equal(
            [],
            test_obj.ids({
                Cell.MTYPE: {'$regex': 'BP'}, })
        )
        # tentative support for 'regex:' prefix
        npt.assert_equal(
            [1, 2],
            test_obj.ids({
                Cell.MTYPE: 'regex:.*BP', })
        )
        # '$regex' is the only query modifier supported for the moment
        with pytest.raises(BluepySnapError):
            test_obj.ids({Cell.MTYPE: {'err': '.*BP'}})

    def test_get(self):
        _call = self.test_obj.get
        assert _call().shape == (3, 12)
        assert _call(0, Cell.MTYPE) == 'L2_X'
        assert _call(np.int32(0), Cell.MTYPE) == 'L2_X'
        pdt.assert_frame_equal(
            _call([1, 2], properties=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT]),
            pd.DataFrame(
                [
                    [201., 'L6_Y', 0.2],
                    [301., 'L6_Y', 0.3],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
                index=[1, 2]
            )
        )
        pdt.assert_frame_equal(
            _call("Node12_L6_Y", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]),
            pd.DataFrame(
                [
                    [201., 'L6_Y', 6],
                    [301., 'L6_Y', 6],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.LAYER],
                index=[1, 2]
            )
        )

        assert _call("Node0_L6_Y", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]).empty
        with pytest.raises(BluepySnapError):
            _call(0, properties='no-such-property')
        with pytest.raises(BluepySnapError):
            _call(999)  # invalid node id
        with pytest.raises(BluepySnapError):
            _call([0, 999])  # one of node ids is invalid

    def test_get_with_library(self):
        test_obj = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_with_library.h5'),
            "default")
        assert test_obj.property_names == {"categorical", "string", "int", "float"}
        res = test_obj.get(properties=["categorical", "string", "int", "float"])
        assert is_categorical(res["categorical"])
        assert res["categorical"].tolist() == ['A', 'A', 'B', 'A']
        assert res["categorical"].cat.categories.tolist() == ['A', 'B', 'C']
        assert res["categorical"].cat.codes.tolist() == [0, 0, 1, 0]
        assert res["string"].tolist() == ["AA", "BB", "CC", "DD"]
        assert res["int"].tolist() == [0, 0, 1, 0]
        npt.assert_allclose(res["float"].tolist(), [0., 0., 1.1, 0.])

    def test_positions(self):
        _call = self.test_obj.positions
        pdt.assert_series_equal(
            _call(0),
            pd.Series([101., 102., 103.], index=[Cell.X, Cell.Y, Cell.Z], name=0)
        )
        pdt.assert_frame_equal(
            _call([2, 0]),
            pd.DataFrame([
                [301., 302., 303.],
                [101., 102., 103.],
            ], index=[2, 0], columns=[Cell.X, Cell.Y, Cell.Z])
        )

    def test_orientations(self):
        _call = self.test_obj.orientations
        npt.assert_almost_equal(
            _call(0),
            [
                [0.738219, 0., 0.674560],
                [0., 1., 0.],
                [-0.674560, 0., 0.738219],
            ],
            decimal=6
        )
        pdt.assert_series_equal(
            _call([2, 0, 1]),
            pd.Series(
                [
                    np.array([
                        [0.462986, 0., 0.886365],
                        [0., 1., 0.],
                        [-0.886365, 0., 0.462986],
                    ]),
                    np.array([
                        [0.738219, 0., 0.674560],
                        [0., 1., 0.],
                        [-0.674560, 0., 0.738219],
                    ]),
                    np.array([
                        [-0.86768965, -0.44169042, 0.22808825],
                        [0.48942842, -0.8393853, 0.23641518],
                        [0.0870316, 0.31676788, 0.94450178]
                    ])
                ],
                index=[2, 0, 1],
                name='orientation'
            )
        )

        # NodePopulation without rotation_angle[x|z]
        _call_no_xz = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_no_xz_rotation.h5'),
            "default").orientations
        # 0 and 2 node_ids have x|z rotation angles equal to zero
        npt.assert_almost_equal(_call_no_xz(0), _call(0))
        npt.assert_almost_equal(_call_no_xz(2), _call(2))
        npt.assert_almost_equal(
            _call_no_xz(1),
            [
                [0.97364046, - 0., 0.22808825],
                [0., 1., - 0.],
                [-0.22808825, 0., 0.97364046]
            ],
            decimal=6
        )

        # NodePopulation without rotation_angle
        _call_no_rot = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_no_rotation.h5'),
            "default").orientations

        pdt.assert_series_equal(
            _call_no_rot([2, 0, 1]),
            pd.Series(
                [np.eye(3), np.eye(3), np.eye(3)],
                index=[2, 0, 1],
                name='orientation'
            )
        )

        # NodePopulation with quaternions
        _call_quat = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_quaternions.h5'),
            "default").orientations

        npt.assert_almost_equal(
            _call_quat(0),
            [
                [1, 0., 0.],
                [0., 0, -1.],
                [0., 1., 0],
            ],
            decimal=6
        )

        series = _call_quat([2, 0, 1])
        for i in range(len(series)):
            series.iloc[i] = np.around(series.iloc[i], decimals=1).astype(np.float64)

        pdt.assert_series_equal(
            series,
            pd.Series(
                [
                    np.array([
                        [0., -1., 0.],
                        [1., 0., 0.],
                        [0., 0., 1.],
                    ]),
                    np.array([
                        [1., 0., 0.],
                        [0., 0., -1.],
                        [0., 1., 0.],
                    ]),
                    np.array([
                        [0., 0., 1.],
                        [0., 1., 0.],
                        [-1., 0., 0.],
                    ])
                ],
                index=[2, 0, 1],
                name='orientation'
            )
        )

        _call_missing_quat = TestNodePopulation.create_population(
            str(TEST_DATA_DIR / 'nodes_quaternions_w_missing.h5'),
            "default").orientations

        with pytest.raises(BluepySnapError):
            _call_missing_quat(0)

    def test_count(self):
        _call = self.test_obj.count
        assert _call(0) == 1
        assert _call([0, 1]) == 2
        assert _call({Cell.MTYPE: 'L6_Y'}) == 2
        assert _call('Layer23') == 1

    def test_morph(self):
        from bluepysnap.morph import MorphHelper
        self.test_obj._node_storage.circuit.config = {
            'components': {
                'morphologies_dir': 'test'
            }
        }
        assert isinstance(self.test_obj.morph, MorphHelper)
