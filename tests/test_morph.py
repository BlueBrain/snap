import h5py
import six
import numpy as np
import numpy.testing as npt
import pandas as pd

from mock import Mock, patch
import pandas.testing as pdt
import pytest

import bluepysnap.morph as test_module
from bluepysnap.circuit import Circuit
from bluepysnap.nodes import NodeStorage
from bluepysnap.sonata_constants import Node
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR, copy_circuit, edit_config


class TestMorphHelper(object):

    @staticmethod
    def create_population(filepath, pop_name):
        config = {
            'nodes_file': filepath,
            'node_types_file': None,
        }
        circuit = Mock()
        return NodeStorage(config, circuit).population(pop_name)

    def setup(self):
        self.nodes = self.create_population(
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
                dt = h5py.special_dtype(vlen=six.text_type)
                h5f.create_dataset('nodes/default/0/@library/model_type',
                                   data=np.array(["biophysical", ], dtype=object), dtype=dt)

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
                dt = h5py.special_dtype(vlen=six.text_type)
                h5f.create_dataset('nodes/default/0/@library/model_type',
                                   data=np.array(["virtual", ], dtype=object), dtype=dt)

            with pytest.raises(BluepySnapError):
                circuit = Circuit(str(config_copy_path))
                circuit.nodes['default'].morph

    def test_get_filepath(self):
        node_id = 0
        assert self.nodes.get(node_id, properties="morphology") == "morph-A"
        actual = self.test_obj.get_filepath(node_id)
        expected = str(self.morph_path / 'morph-A.swc')
        assert actual == expected

    def test_get_1(self):
        actual = self.test_obj.get(0).points
        assert len(actual) == 32
        expected = [
            [-0.32, 1.0, 0., 0.725],
            [-0.32, 0.9, 0., 0.820],
        ]
        npt.assert_almost_equal(expected, actual[:2])

    def test_get_2(self):
        node_id = 0
        actual = self.test_obj.get(node_id, transform=True).points

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
        assert len(actual) == 32
        # swc file
        # index       type         X            Y            Z       radius       parent
        # 1           1    -0.320000     1.000000     0.000000     0.725000          -1
        # 2           1    -0.320000     0.900000     0.000000     0.820000           1

        # rotation around the x axis 90 degrees counter clockwise
        expected = [
            [100.68, 102, 104.0, 0.725],
            [100.68, 102, 103.9, 0.820]
        ]
        npt.assert_almost_equal(
            expected, actual[:2]
        )
        npt.assert_almost_equal(expected, actual[:2])

    def test_get_3(self):
        nodes = self.create_population(
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

        assert len(actual) == 32
        expected = [
            [100.7637696, 103, 103.2158592, 0.725],
            [100.7637696, 102.9, 103.2158592, 0.820]
        ]
        npt.assert_almost_equal(expected, actual[:2])


@patch(test_module.__name__ + '.MORPH_CACHE_SIZE', 1)
@patch('neurom.load_neuron')
def test_MorphHelper_cache_1(nm_load):
    nodes = Mock()
    nodes.get.side_effect = ['morph-A', 'morph-A', 'morph-B', 'morph-A']
    with patch.object(test_module.MorphHelper, '_is_biophysical', return_value=True):
        test_obj = test_module.MorphHelper('morph-dir', nodes)
        nm_load.side_effect = Mock
        morph1 = test_obj.get(1)
        # should get cached object for 'morph-A'
        assert test_obj.get(2) is morph1
        # should get new object ('morph-B')
        assert test_obj.get(3) is not morph1
        # 'morph-A' was evicted from cache
        assert test_obj.get(4) is not morph1


@patch(test_module.__name__ + '.MORPH_CACHE_SIZE', None)
@patch('neurom.load_neuron')
def test_MorphHelper_cache_2(nm_load):
    nodes = Mock()
    nodes.get.side_effect = ['morph-A', 'morph-A']
    with patch.object(test_module.MorphHelper, '_is_biophysical', return_value=True):
        test_obj = test_module.MorphHelper('morph-dir', nodes)
        nm_load.side_effect = Mock
        morph1 = test_obj.get(1)
        # same morphology, but no caching
        assert test_obj.get(2) is not morph1
