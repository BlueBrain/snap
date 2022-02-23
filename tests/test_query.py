import numpy.testing as npt
import pandas as pd
import pytest

from bluepysnap import BluepySnapError
from bluepysnap.query import _circuit_mask, _positional_mask, resolve_ids


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

    queries, mask = _circuit_mask(
        data, "default", {"population": "default", "node_id": [2], "other": "val"}
    )
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [False, False, True])

    queries, mask = _circuit_mask(data, "default", {"other": "val"})
    assert queries == {"other": "val"}
    npt.assert_array_equal(mask, [True, True, True])


def test_resolve_ids():
    data = pd.DataFrame(
        [[1, 0.4, "seven"], [2, 0.5, "eight"], [3, 0.6, "nine"]], columns=["int", "float", "str"]
    )
    assert [False, True, False] == resolve_ids(data, "", {"str": "eight"}).tolist()
    assert [False, False, False] == resolve_ids(data, "", {"int": 1, "str": "eight"}).tolist()
    assert [True, True, False] == resolve_ids(
        data, "", {"$or": [{"str": "seven"}, {"float": (0.5, 0.5)}]}
    ).tolist()
    assert [False, False, True] == resolve_ids(
        data, "", {"$and": [{"str": "nine"}, {"int": 3}]}
    ).tolist()
    assert [False, False, True] == resolve_ids(
        data, "", {"$and": [{"str": "nine"}, {"$or": [{"int": 1}, {"int": 3}]}]}
    ).tolist()
    assert [False, True, True] == resolve_ids(
        data, "", {"$or": [{"float": (0.59, 0.61)}, {"$and": [{"str": "eight"}, {"int": 2}]}]}
    ).tolist()
    assert [True, False, True] == resolve_ids(
        data, "", {"$or": [{"node_id": 0}, {"edge_id": 2}]}
    ).tolist()

    with pytest.raises(BluepySnapError) as e:
        resolve_ids(data, "", {"str": {"$regex": "*.some", "edge_id": 2}})
    assert "Value operators can't be used with plain values" in e.value.args[0]
