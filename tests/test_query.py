import pandas as pd
import numpy.testing as npt
import pytest

from bluepysnap import BluepySnapError
from bluepysnap.query import operator_mask, _positional_mask, _circuit_mask


def test_positional_mask():
    data = pd.DataFrame(range(3))
    npt.assert_array_equal(_positional_mask(data, [1, 2]), [False, True, True])
    npt.assert_array_equal(_positional_mask(data, [0, 2]), [True, False, True])


def test_population_mask():
    data = pd.DataFrame(range(3))
    queries, mask = _circuit_mask(data, "default", {"population": "default", "other": "val"})
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [True, True, True])

    queries, mask = _circuit_mask(data, "default", {"population": "unknown", "other": "val"})
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [False, False, False])

    queries, mask = _circuit_mask(data, "default",
                                  {"population": "default", "node_id": [2], "other": "val"})
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [False, False, True])

    queries, mask = _circuit_mask(data, "default", {"other": "val"})
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [True, True, True])


def test_operator_mask():
    data = pd.DataFrame([
        [1, .4, 'seven'],
        [2, .5, 'eight'],
        [3, .6, 'nine']],
        columns=['int', 'float', 'str'])
    assert [False, True, False] == operator_mask(data, '', {'str': 'eight'}).tolist()
    assert [False, False, False] == operator_mask(data, '', {'int': 1, 'str': 'eight'}).tolist()
    assert [True, True, False] == operator_mask(
        data, '', {'$or': [{'str': 'seven'}, {'float': (.5, .5)}]}).tolist()
    assert [False, False, True] == operator_mask(
        data, '', {'$and': [{'str': 'nine'}, {'int': 3}]}).tolist()
    assert [False, False, True] == operator_mask(
        data, '', {'$and': [{'str': 'nine'}, {'$or': [{'int': 1}, {'int': 3}]}]}).tolist()
    assert [True, False, True] == operator_mask(
        data, '', {'$or': [{'node_id': 0}, {'edge_id': 2}]}).tolist()

    with pytest.raises(BluepySnapError) as e:
        operator_mask(data, '', {'str': {'$regex': '*.some', 'edge_id': 2}})
    assert "Value operators can't be used with plain values" in e.value.args[0]
