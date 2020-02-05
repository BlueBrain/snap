import numpy as np
import numpy.testing as npt
import pandas as pd

from mock import Mock, patch

import bluepysnap.morph as test_module

from utils import TEST_DATA_DIR


class TestMorphHelper(object):
    def setup(self):
        self.nodes = Mock()
        self.morph_path = TEST_DATA_DIR / 'morphologies'
        self.test_obj = test_module.MorphHelper(str(self.morph_path), self.nodes)

    def test_get_filepath(self):
        self.nodes.get.return_value = 'test'
        actual = self.test_obj.get_filepath(42)
        expected = str(self.morph_path / 'test.swc')
        assert actual == expected

    def test_get_1(self):
        self.nodes.get.return_value = 'small_morph'
        actual = self.test_obj.get(42).points
        assert len(actual) == 32
        expected = [
            [-0.32, 1.0, 0., 0.725],
            [-0.32, 0.9, 0., 0.820],
        ]
        npt.assert_almost_equal(expected, actual[:2])

    def test_get_2(self):
        self.nodes.get.return_value = 'small_morph'
        self.nodes.positions.return_value = pd.Series({
            'x': 100.0,
            'y': 200.0,
            'z': 300.0,
        })
        self.nodes.orientations.return_value = -np.identity(3)
        actual = self.test_obj.get(42, transform=True).points
        assert len(actual) == 32
        expected = [
            [100.32, 199.0, 300., 0.725],
            [100.32, 199.1, 300., 0.820]
        ]
        npt.assert_almost_equal(expected, actual[:2])


@patch(test_module.__name__ + '.MORPH_CACHE_SIZE', 1)
@patch('neurom.load_neuron')
def test_MorphHelper_cache_1(nm_load):
    nodes = Mock()
    nodes.get.side_effect = ['morph-A', 'morph-A', 'morph-B', 'morph-A']
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
    test_obj = test_module.MorphHelper('morph-dir', nodes)
    nm_load.side_effect = Mock
    morph1 = test_obj.get(1)
    # same morphology, but no no caching
    assert test_obj.get(2) is not morph1