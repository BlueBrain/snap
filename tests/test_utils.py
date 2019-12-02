import os

import numpy as np
import numpy.testing as npt
import pytest

import bluepysnap.utils as test_module
from bluepysnap.exceptions import BluepySnapError


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_load_json():
    actual = test_module.load_json(os.path.join(TEST_DATA_DIR, 'circuit_config.json'))
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


def test_euler2mat():
    pi2 = 0.5 * np.pi
    actual = test_module.euler2mat(
        [0.0, pi2],  # rotation_angle_z
        [pi2, 0.0],  # rotation_angle_y
        [pi2, pi2],  # rotation_angle_x
    )
    expected = np.array([
        [
            [ 0.,  0.,  1.],
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
        ],
        [
            [ 0., -1.,  0.],
            [ 0.,  0., -1.],
            [ 1.,  0.,  0.],
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
