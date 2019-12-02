import os

import mock
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.util.testing as pdt
import pytest

from mock import patch, Mock

from bluepysnap.bbp import Cell
from bluepysnap.exceptions import BluepySnapError

import bluepysnap.nodes as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_get_population_name_duplicate():
    storage = Mock()
    storage.population_names = ['a', 'b']
    with patch(test_module.__name__ + '.libsonata.NodeStorage') as NodeStorage:
        NodeStorage.return_value = storage
        with pytest.raises(BluepySnapError):
            test_module._get_population_name(mock.ANY)


def test_node_ids_by_filter():
    nodes = pd.DataFrame({
        Cell.X: [0.0, 0.5, 1.0],
        Cell.MTYPE: pd.Categorical.from_codes([0, 1, 1], ['A', 'B', 'C']),
        Cell.LAYER: [1, 2, 3],
    })
    npt.assert_equal(
        [],
        test_module._node_ids_by_filter(nodes, {Cell.MTYPE: 'err'})
    )
    npt.assert_equal(
        [1],
        test_module._node_ids_by_filter(nodes, {
            Cell.X: (0, 0.7),
            Cell.MTYPE: ['B', 'C'],
            Cell.LAYER: (1, 2)
        })
    )
    with pytest.raises(BluepySnapError):
        test_module._node_ids_by_filter(nodes, {'err': 23})


def test_node_ids_by_filter_complex_query():
    nodes = pd.DataFrame({
        Cell.MTYPE: ['L23_MC', 'L4_BP', 'L6_BP', 'L6_BPC'],
    })
    # only full match is accepted
    npt.assert_equal(
        [1, 2],
        test_module._node_ids_by_filter(nodes, {
            Cell.MTYPE: {'$regex': '.*BP'},
        })
    )
    # ...not 'startswith'
    npt.assert_equal(
        [],
        test_module._node_ids_by_filter(nodes, {
            Cell.MTYPE: {'$regex': 'L6'},
        })
    )
    # ...or 'endswith'
    npt.assert_equal(
        [],
        test_module._node_ids_by_filter(nodes, {
            Cell.MTYPE: {'$regex': 'BP'},
        })
    )
    # tentative support for 'regex:' prefix
    npt.assert_equal(
        [1, 2],
        test_module._node_ids_by_filter(nodes, {
            Cell.MTYPE: 'regex:.*BP',
        })
    )
    # '$regex' is the only query modifier supported for the moment
    with pytest.raises(BluepySnapError):
        test_module._node_ids_by_filter(nodes, {Cell.MTYPE: {'err': '.*BP'}})


class TestNodePopulation:
    def setup(self):
        config = {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes.h5'),
            'node_types_file': None,
            'node_sets_file': os.path.join(TEST_DATA_DIR, 'node_sets.json'),
        }
        circuit = Mock()
        self.test_obj = test_module.NodePopulation(config, circuit)

    def test_basic(self):
        assert self.test_obj.name == 'default'
        assert self.test_obj.size == 3
        assert(
            sorted(self.test_obj.property_names) ==
            [
                Cell.HOLDING_CURRENT,
                Cell.LAYER,
                Cell.MORPHOLOGY,
                Cell.MTYPE,
                'rotation_angle_xaxis',
                'rotation_angle_yaxis',
                'rotation_angle_zaxis',
                'x',
                'y',
                'z',
            ]
        )
        assert(
            sorted(self.test_obj.node_sets) ==
            ['Empty', 'EmptyDict', 'Empty_L6_Y', 'Failing', 'Layer2', 'Layer23', 'Node012',
             'Node0_L6_Y', 'Node122', 'Node12_L6_Y', 'Node2_L6_Y']
        )

    def test_property_values(self):
        assert(
            self.test_obj.property_values(Cell.LAYER) ==
            {2, 6}
        )
        assert(
            self.test_obj.property_values(Cell.MORPHOLOGY) ==
            {'morph-A', 'morph-B', 'morph-C'}
        )

    def test_ids(self):
        _call = self.test_obj.ids
        npt.assert_equal(_call(), [0, 1, 2])
        npt.assert_equal(_call(limit=1), [0])
        npt.assert_equal(len(_call(sample=2)), 2)
        npt.assert_equal(_call(0), [0])
        npt.assert_equal(_call([0, 1]), [0, 1])
        npt.assert_equal(_call([1, 0, 1]), [1, 0, 1])  # order and duplicates preserved
        npt.assert_equal(_call(np.array([1, 0, 1])), np.array([1, 0, 1]))
        npt.assert_equal(_call({Cell.MTYPE: 'L6_Y'}), [1, 2])
        npt.assert_equal(_call('Layer23'), [0])
        npt.assert_equal(_call('Node012'), [0, 1, 2])  # order preserved
        npt.assert_equal(_call('Node122'), [1, 2, 2])  # order and duplicates preserved
        npt.assert_equal(_call('Node12_L6_Y'), [1, 2])
        npt.assert_equal(_call('Node2_L6_Y'), [2])
        npt.assert_equal(_call('Node0_L6_Y'), [])  # return empty if disjoint samples
        npt.assert_equal(_call('Empty'), [])  # return empty if empty node_id = []
        npt.assert_equal(_call('Empty_L6_Y'), [])  # return empty if empty node_id = []
        npt.assert_equal(_call('EmptyDict'), _call())  # return all ids

        with pytest.raises(BluepySnapError):
            _call('no-such-node-set')
        with pytest.raises(BluepySnapError):
            _call('Failing')
        with pytest.raises(BluepySnapError):
            _call(-1)  # node ID out of range (lower boundary)
        with pytest.raises(BluepySnapError):
            _call(999)  #  node ID out of range (upper boundary)
        with pytest.raises(BluepySnapError):
            _call([1, 999])  # one of node IDs out of range

    def test_get(self):
        _call = self.test_obj.get
        assert _call().shape == (3, 10)
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
        pdt.assert_frame_equal(
            _call("EmptyDict", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]),
            pd.DataFrame(
                [
                    [101., 'L2_X', 2],
                    [201., 'L6_Y', 6],
                    [301., 'L6_Y', 6],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.LAYER],
                index=[0, 1, 2]
            )
        )
        assert _call("Node0_L6_Y", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]).empty
        with pytest.raises(BluepySnapError):
            _call(0, properties='no-such-property')
        with pytest.raises(BluepySnapError):
            _call(999)  # invalid node id
        with pytest.raises(BluepySnapError):
            _call([0, 999])  # one of node ids is invalid

    def test_positions(self):
        _call = self.test_obj.positions
        pdt.assert_series_equal(
            _call(0),
            pd.Series([101., 102., 103.], index=list('xyz'), name=0)
        )
        pdt.assert_frame_equal(
            _call([2, 0]),
            pd.DataFrame([
                [301., 302., 303.],
                [101., 102., 103.],
            ], index=[2, 0], columns=list('xyz'))
        )

    def test_orientations(self):
        _call = self.test_obj.orientations
        npt.assert_almost_equal(
            _call(0),
            [
                [ 0.738219, 0.,  0.674560],
                [ 0.     ,  1.,  0.      ],
                [-0.674560, 0.,  0.738219],
            ],
            decimal=6
        )
        pdt.assert_series_equal(
            _call([2, 0, 1]),
            pd.Series(
                [
                    np.array([
                        [ 0.462986, 0.,  0.886365],
                        [ 0.      , 1.,  0.      ],
                        [-0.886365, 0.,  0.462986],
                    ]),
                    np.array([
                        [ 0.738219, 0.,  0.674560],
                        [ 0.      , 1.,  0.      ],
                        [-0.674560, 0.,  0.738219],
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
        config = {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes_no_xz_rotation.h5'),
            'node_types_file': None,
        }
        circuit = Mock()
        _call_no_xz = test_module.NodePopulation(config, circuit).orientations
        # 0 and 2 node_ids have x|z rotation angles equal to zero
        npt.assert_almost_equal(_call_no_xz(0), _call(0))
        npt.assert_almost_equal(_call_no_xz(2), _call(2))
        npt.assert_almost_equal(
            _call_no_xz(1),
            [
                [0.97364046, - 0., 0.22808825],
                [0.        ,   1., - 0.      ],
                [-0.22808825,  0., 0.97364046]
            ],
            decimal=6
        )

        # NodePopulation without rotation_angle
        config = {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes_no_rotation.h5'),
            'node_types_file': None,
        }
        circuit = Mock()
        _call_no_rot = test_module.NodePopulation(config, circuit).orientations
        pdt.assert_series_equal(
            _call_no_rot([2, 0, 1]),
            pd.Series(
                [np.eye(3), np.eye(3), np.eye(3)],
                index=[2, 0, 1],
                name='orientation'
            )
        )

        # NodePopulation with quaternions
        config = {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes_quaternions.h5'),
            'node_types_file': None,
        }
        circuit = Mock()
        _call_quat = test_module.NodePopulation(config, circuit).orientations
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

    config = {
        'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes_quaternions_w_missing.h5'),
        'node_types_file': None,
    }
    circuit = Mock()
    _call_missing_quat = test_module.NodePopulation(config, circuit).orientations
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
        self.test_obj._circuit.config = {
            'components': {
                'morphologies_dir': 'test'
            }
        }
        assert isinstance(self.test_obj.morph, MorphHelper)


def test_NodePopulation_without_node_sets():
    config = {
        'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes.h5'),
        'node_types_file': None,
    }
    nodes = test_module.NodePopulation(config, mock.ANY)
    assert nodes.node_sets == {}
