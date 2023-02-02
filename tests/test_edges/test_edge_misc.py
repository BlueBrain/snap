from unittest.mock import ANY, Mock

import numpy as np
import numpy.testing as npt
import pytest

import bluepysnap.edges.edge_population as test_module


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
        test_module._estimate_range_size(ANY, [])
