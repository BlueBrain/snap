import numpy as np
import numpy.testing as npt
import pytest

import bluepysnap.utils as test_module
from bluepysnap.sonata_constants import DYNAMICS_PREFIX
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR


def test_load_json():
    actual = test_module.load_json(str(TEST_DATA_DIR / 'circuit_config.json'))
    assert actual['manifest']['$BASE_DIR'] == '.'


def test_is_iterable():
    assert test_module.is_iterable([12, 13])
    assert test_module.is_iterable(np.asarray([12, 13]))
    assert not test_module.is_iterable(12)
    assert not test_module.is_iterable('abc')


def test_ensure_list():
    assert test_module.ensure_list(1) == [1]
    assert test_module.ensure_list([1]) == [1]
    assert test_module.ensure_list(iter([1])) == [1]
    assert test_module.ensure_list((2, 1)) == [2, 1]
    assert test_module.ensure_list('abc') == ['abc']


def test_roundrobin():
    a = [[1, 2, 3], [4], [5, 6]]
    assert list(test_module.roundrobin(*a)) == [1, 4, 5, 2, 6, 3]


def test_fix_libsonata_empty_list():
    npt.assert_array_equal(test_module.fix_libsonata_empty_list(), np.array([-2]))


def test_add_dynamic_prefix():
    assert test_module.add_dynamic_prefix(["a", "b"]) == [DYNAMICS_PREFIX + "a",
                                                          DYNAMICS_PREFIX + "b"]


def test_euler2mat():
    pi2 = 0.5 * np.pi
    actual = test_module.euler2mat(
        [0.0, pi2],  # rotation_angle_z
        [pi2, 0.0],  # rotation_angle_y
        [pi2, pi2],  # rotation_angle_x
    )
    expected = np.array([
        [
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ],
        [
            [0., -1., 0.],
            [0., 0., -1.],
            [1., 0., 0.],
        ],
    ])
    npt.assert_almost_equal(actual, expected)

    with pytest.raises(BluepySnapError):
        test_module.euler2mat([pi2, pi2], [pi2, pi2], [pi2])  # ax|y|z not of same size


def test_quaternion2mat():
    actual = test_module.quaternion2mat([1, 1, 1], [1, 0, 0, ], [0, 1, 0], [0, 0, 1])
    expected = np.array([
        [
            [1., 0., 0.],
            [0., 0., -1.],
            [0., 1., 0.],
        ],
        [
            [0., 0., 1.],
            [0., 1., 0.],
            [-1., 0., 0.],
        ],
        [
            [0., -1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ],
    ])
    npt.assert_almost_equal(actual, expected)
