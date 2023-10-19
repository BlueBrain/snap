import numpy as np
import numpy.testing as npt
import pytest

import bluepysnap.utils as test_module
from bluepysnap.circuit_ids_types import CircuitEdgeId, CircuitNodeId
from bluepysnap.exceptions import (
    BluepySnapDeprecationError,
    BluepySnapDeprecationWarning,
    BluepySnapError,
)
from bluepysnap.sonata_constants import DYNAMICS_PREFIX

from utils import TEST_DATA_DIR


def test_load_json():
    actual = test_module.load_json(str(TEST_DATA_DIR / "circuit_config.json"))
    assert actual["manifest"]["$BASE_DIR"] == "."


def test_is_iterable():
    assert test_module.is_iterable([12, 13])
    assert test_module.is_iterable(np.asarray([12, 13]))
    assert not test_module.is_iterable(12)
    assert not test_module.is_iterable("abc")


def test_is_node_id():
    assert test_module.is_node_id(1)
    assert test_module.is_node_id(np.int32(1))
    assert test_module.is_node_id(np.uint32(1))
    assert test_module.is_node_id(np.int64(1))
    assert test_module.is_node_id(np.uint64(1))
    assert test_module.is_node_id(CircuitNodeId("default", 1))

    assert not test_module.is_node_id([1])
    assert not test_module.is_node_id(np.array([1], dtype=np.int64))
    assert not test_module.is_node_id(CircuitEdgeId("default", 1))


def test_ensure_list():
    assert test_module.ensure_list(1) == [1]
    assert test_module.ensure_list([1]) == [1]
    assert test_module.ensure_list(iter([1])) == [1]
    assert test_module.ensure_list((2, 1)) == [2, 1]
    assert test_module.ensure_list("abc") == ["abc"]


def test_ensure_ids():
    res = test_module.ensure_ids(np.array([1, 2, 3], dtype=np.uint64))
    npt.assert_equal(res, np.array([1, 2, 3], dtype=test_module.IDS_DTYPE))
    npt.assert_equal(res.dtype, test_module.IDS_DTYPE)


def test_add_dynamic_prefix():
    assert test_module.add_dynamic_prefix(["a", "b"]) == [
        DYNAMICS_PREFIX + "a",
        DYNAMICS_PREFIX + "b",
    ]


def test_euler2mat():
    pi2 = 0.5 * np.pi
    actual = test_module.euler2mat(
        [0.0, pi2],  # rotation_angle_z
        [pi2, 0.0],  # rotation_angle_y
        [pi2, pi2],  # rotation_angle_x
    )
    expected = np.array(
        [
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    npt.assert_almost_equal(actual, expected)

    with pytest.raises(BluepySnapError):
        test_module.euler2mat([pi2, pi2], [pi2, pi2], [pi2])  # ax|y|z not of same size


def test_quaternion2mat():
    actual = test_module.quaternion2mat(
        [1, 1, 1],
        [
            1,
            0,
            0,
        ],
        [0, 1, 0],
        [0, 0, 1],
    )
    expected = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        ]
    )
    npt.assert_almost_equal(actual, expected)


class TestDeprecate:
    def test_fail(self):
        with pytest.raises(BluepySnapDeprecationError):
            test_module.Deprecate.fail("something")

    def test_warning(self):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            test_module.Deprecate.warn("something")

            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, BluepySnapDeprecationWarning)
            assert "something" in str(w[-1].message)
