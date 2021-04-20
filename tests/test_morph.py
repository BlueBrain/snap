import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd

from mock import Mock, patch
import pandas.testing as pdt
import pytest

import bluepysnap.morph as test_module
from bluepysnap.circuit import Circuit
from bluepysnap.sonata_constants import Node
from bluepysnap.circuit_ids import CircuitNodeId
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR, copy_circuit, edit_config, create_node_population


class TestMorphHelper:

    def setup(self):
        self.nodes = create_node_population(
            str(TEST_DATA_DIR / 'nodes_quaternions.h5'),
            "default")
        self.morph_path = TEST_DATA_DIR / 'morphologies'
        self.test_obj = test_module.MorphHelper(str(self.morph_path), self.nodes)

    def test_biophysical_in_library(self):
        with copy_circuit() as (circuit_copy_path, config_copy_path):
            with edit_config(config_copy_path) as config:
                config["networks"]["nodes"][0]["nodes_file"] = "$NETWORK_DIR/nodes_quaternions.h5"
            nodes_file = circuit_copy_path / 'nodes_quaternions.h5'
            with h5py.File(nodes_file, 'r+') as h5f:
                data = h5f['nodes/default/0/model_type'][:]
                del h5f['nodes/default/0/model_type']
                h5f.create_dataset('nodes/default/0/model_type',
                                   data=np.zeros_like(data, dtype=int))
                h5f.create_dataset('nodes/default/0/@library/model_type',
                                   data=np.array(["biophysical", ], dtype=h5py.string_dtype()))

            circuit = Circuit(str(config_copy_path))
            assert isinstance(circuit.nodes['default'].morph,  test_module.MorphHelper)

    def test_not_biophysical_population(self):
        with copy_circuit() as (circuit_copy_path, config_copy_path):
            with edit_config(config_copy_path) as config:
                config["networks"]["nodes"][0]["nodes_file"] = "$NETWORK_DIR/nodes_quaternions.h5"
            nodes_file = circuit_copy_path / 'nodes_quaternions.h5'
            with h5py.File(nodes_file, 'r+') as h5f:
                data = h5f['nodes/default/0/model_type'][:]
                del h5f['nodes/default/0/model_type']
                h5f.create_dataset('nodes/default/0/model_type',
                                   data=np.zeros_like(data, dtype=int))
                h5f.create_dataset('nodes/default/0/@library/model_type',
                                   data=np.array(["virtual", ], dtype=h5py.string_dtype()))

            with pytest.raises(BluepySnapError):
                circuit = Circuit(str(config_copy_path))
                circuit.nodes['default'].morph

    def test_get_filepath(self):
        node_id = 0
        assert self.nodes.get(node_id, properties="morphology") == "morph-A"
        actual = self.test_obj.get_filepath(node_id)
        expected = self.morph_path / 'morph-A.swc'
        assert actual == expected
        node_id = CircuitNodeId("default", 0)
        assert self.nodes.get(node_id, properties="morphology") == "morph-A"
        actual = self.test_obj.get_filepath(node_id)
        assert actual == expected

        with pytest.raises(BluepySnapError):
            self.test_obj.get_filepath([CircuitNodeId("default", 0), CircuitNodeId("default", 1)])

        with pytest.raises(BluepySnapError):
            self.test_obj.get_filepath([0, 1])

    def test_get_morphology(self):
        actual = self.test_obj.get(0).points
        assert len(actual) == 13
        expected = [
            [0., 5., 0., 1.],
            [2., 9., 0., 1.],
        ]
        npt.assert_almost_equal(expected, actual[:2])

        with pytest.raises(BluepySnapError):
            self.test_obj.get([0, 1])

    def test_get_morphology_simple_rotation(self):
        node_id = 0
        # check that the input node positions / orientation values are still the same
        pdt.assert_series_equal(self.nodes.positions(node_id),
                                pd.Series([101., 102., 103.],
                                          index=[Node.X, Node.Y, Node.Z], name=0))
        npt.assert_almost_equal(
            self.nodes.orientations(node_id),
            [
                [1,  0., 0.],
                [0., 0., -1.],
                [0., 1., 0.],
            ],
            decimal=6
        )

        actual = self.test_obj.get(node_id, transform=True).points
        assert len(actual) == 13
        # swc file
        # index       type         X            Y            Z       radius       parent
        #   22           2     0.000000     5.000000     0.000000     1.000000           1
        #   23           2     2.000000     9.000000     0.000000     1.000000          22
        # rotation around the x axis 90 degrees counter clockwise (swap Y and Z)
        # x = X + 101, y = Z + 102, z = Y + 103, radius does not change
        expected = [
            [101., 102., 108.,   1.],
            [103., 102., 112.,   1.]
        ]
        npt.assert_almost_equal(actual[:2], expected)

    def test_get_morphology_standard_rotation(self):
        nodes = create_node_population(
            str(TEST_DATA_DIR / 'nodes.h5'),
            "default")
        test_obj = test_module.MorphHelper(str(self.morph_path), nodes)

        node_id = 0
        actual = test_obj.get(node_id, transform=True).points

        # check if the input node positions / orientation values are still the same
        pdt.assert_series_equal(self.nodes.positions(node_id),
                                pd.Series([101., 102., 103.],
                                          index=[Node.X, Node.Y, Node.Z], name=0))
        npt.assert_almost_equal(
            nodes.orientations(node_id),
            [
                [0.738219, 0., 0.674560],
                [0., 1., 0.],
                [-0.674560, 0., 0.738219],
            ],
            decimal=6
        )

        assert len(actual) == 13
        expected = [
            [101., 107., 103., 1.],
            [102.47644, 111., 101.65088, 1.]
        ]
        npt.assert_almost_equal(actual[:2], expected, decimal=6)


@patch(test_module.__name__ + '.MORPH_CACHE_SIZE', 1)
@patch('neurom.load_neuron')
def test_MorphHelper_cache_1(nm_load):
    nodes = Mock()
    nodes.get.side_effect = ['morph-A', 'morph-A', 'morph-B', 'morph-A']
    with patch.object(test_module.MorphHelper, '_is_biophysical', return_value=True):
        test_obj = test_module.MorphHelper('morph-dir', nodes)
        nm_load.side_effect = Mock
        morph0 = test_obj.get(0)
        # should get cached object for 'morph-A'
        assert test_obj.get(1) is morph0
        # should get new object ('morph-B')
        assert test_obj.get(2) is not morph0
        # 'morph-A' was evicted from cache
        assert test_obj.get(3) is not morph0


@patch(test_module.__name__ + '.MORPH_CACHE_SIZE', None)
@patch('neurom.load_neuron')
def test_MorphHelper_cache_2(nm_load):
    nodes = Mock()
    nodes.get.side_effect = ['morph-A', 'morph-A']
    with patch.object(test_module.MorphHelper, '_is_biophysical', return_value=True):
        test_obj = test_module.MorphHelper('morph-dir', nodes)
        nm_load.side_effect = Mock
        morph1 = test_obj.get(0)
        # same morphology, but no caching
        assert test_obj.get(1) is not morph1
