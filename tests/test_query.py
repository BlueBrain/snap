import pandas as pd
import numpy.testing as npt

from bluepysnap.query import _positional_mask, _circuit_mask


def test_positional_mask():
    data = pd.DataFrame(range(3))
    npt.assert_array_equal(_positional_mask(data, [1, 2]), [False, True, True])
    npt.assert_array_equal(_positional_mask(data, [0, 2]), [True, False, True])


def test_node_population_mask():
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
