from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepysnap.circuit_ids as test_module
from bluepysnap.circuit_ids_types import IDS_DTYPE, CircuitEdgeId, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError

from utils import setup_tempdir


def _multi_index():
    return pd.MultiIndex.from_arrays([["a", "a", "b", "a"], [0, 1, 0, 2]])


class TestCircuitNodeIds:
    @property
    def ids_cls(self):
        return test_module.CircuitNodeIds

    @property
    def id_cls(self):
        return CircuitNodeId

    @property
    def id_name(self):
        return "node_ids"

    def _circuit_ids(self, populations, ids):
        index = pd.MultiIndex.from_arrays([populations, ids])
        index.names = ["population", self.id_name]
        return index

    def setup_method(self):
        self.test_obj_unsorted = self.ids_cls(_multi_index(), sort_index=False)
        self.test_obj_sorted = self.ids_cls(_multi_index())

    def test_init(self):
        values = pd.MultiIndex.from_arrays([["a", "a", "b", "a"], [0, 1, 0, 2]])
        tested = self.ids_cls(values, sort_index=False)
        assert tested.index.names == ["population", self.id_name]
        npt.assert_equal(tested.index.values, values.values)

        tested = self.ids_cls(values, sort_index=True)
        assert tested.index.names == ["population", self.id_name]
        npt.assert_equal(tested.index.values, np.sort(values.values))

        with pytest.raises(BluepySnapError):
            self.ids_cls(1)

        assert isinstance(self.test_obj_sorted, self.ids_cls)

    def test_from_arrays(self):
        tested = self.ids_cls.from_arrays(["a", "b"], [0, 1])
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [0, 1]))

        # keep ids ordering
        tested = self.ids_cls.from_arrays(["a", "b"], [1, 0], sort_index=False)
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [1, 0]))

        # keep population ordering
        tested = self.ids_cls.from_arrays(["b", "a"], [0, 1], sort_index=False)
        pdt.assert_index_equal(tested.index, self._circuit_ids(["b", "a"], [0, 1]))

        # keep duplicates
        tested = self.ids_cls.from_arrays(["a", "a"], [0, 0])
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "a"], [0, 0]))
        assert tested.index.size == 2

        with pytest.raises(BluepySnapError):
            self.ids_cls.from_arrays(["a", "a", "a"], [0, 0])

        with pytest.raises(BluepySnapError):
            self.ids_cls.from_arrays(["a", "a"], [0, 0, 0])

    def test_create_ids(self):
        tested = self.ids_cls.from_dict({"a": [0]})
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a"], [0]))

        tested = self.ids_cls.from_dict({"a": [0, 1]})
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "a"], [0, 1]))

        tested = self.ids_cls.from_dict({"a": [0], "b": [0]})
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [0, 0]))

        tested = self.ids_cls.from_dict({"a": [0], "b": [1]})
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [0, 1]))

        # keep duplicates
        tested = self.ids_cls.from_dict({"a": [0, 0]})
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "a"], [0, 0]))
        assert tested.index.size == 2

    def test_from_tuples(self):
        tested = self.ids_cls.from_tuples([self.id_cls("a", 0), self.id_cls("b", 1)])
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [0, 1]))
        tested = self.ids_cls.from_tuples([("a", 0), ("b", 1)])
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "b"], [0, 1]))

    def test_copy(self):
        tested = self.test_obj_sorted.copy()
        assert self.test_obj_sorted is not tested
        assert self.test_obj_sorted.index is not tested.index
        assert self.test_obj_sorted == tested

    def test_len(self):
        assert len(self.test_obj_sorted) == len(_multi_index())

    def test__locate(self):
        tested = self.test_obj_sorted._locate("a")
        npt.assert_equal(tested, [0, 1, 2])
        tested = self.test_obj_unsorted._locate("a")
        npt.assert_equal(tested, [0, 1, 3])

        tested = self.test_obj_sorted._locate("b")
        npt.assert_equal(tested, [3])
        tested = self.test_obj_unsorted._locate("b")
        npt.assert_equal(tested, [2])

    def test_filter_population(self):
        tested = self.test_obj_sorted.filter_population("a")
        pdt.assert_index_equal(tested.index, self._circuit_ids(["a", "a", "a"], [0, 1, 2]))

        tested = self.test_obj_sorted.filter_population("b")
        pdt.assert_index_equal(tested.index, self._circuit_ids(["b"], [0]))

        tested = self.test_obj_sorted.copy()
        tested.filter_population("b", inplace=True)
        pdt.assert_index_equal(tested.index, self._circuit_ids(["b"], [0]))

    def test_get_populations(self):
        tested = self.test_obj_sorted.get_populations()
        npt.assert_equal(tested, ["a", "a", "a", "b"])
        tested = self.test_obj_unsorted.get_populations()
        npt.assert_equal(tested, ["a", "a", "b", "a"])

        tested = self.test_obj_sorted.get_populations(unique=True)
        npt.assert_equal(tested, ["a", "b"])
        tested = self.test_obj_unsorted.get_populations(unique=True)
        npt.assert_equal(tested, ["a", "b"])

    def test_get_ids(self):
        tested = self.test_obj_sorted.get_ids()
        npt.assert_equal(tested, [0, 1, 2, 0])
        npt.assert_equal(tested.dtype, IDS_DTYPE)
        tested = self.test_obj_unsorted.get_ids()
        npt.assert_equal(tested, [0, 1, 0, 2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj_sorted.get_ids(unique=True)
        npt.assert_equal(tested, [0, 1, 2])
        tested = self.test_obj_unsorted.get_ids(unique=True)
        npt.assert_equal(tested, [0, 1, 2])

    def test_sort(self):
        obj = self.ids_cls(_multi_index(), sort_index=False)
        expected = pd.MultiIndex.from_arrays([["a", "a", "a", "b"], [0, 1, 2, 0]])
        npt.assert_equal(obj.sort().index.values, expected.values)
        obj.sort(inplace=True)
        npt.assert_equal(obj.index.values, expected.values)

    def test_append(self):
        other = self.ids_cls(pd.MultiIndex.from_arrays([["c", "b", "c"], [0, 5, 1]]))
        expected = self.ids_cls(
            self._circuit_ids(["a", "a", "b", "a", "c", "b", "c"], [0, 1, 0, 2, 0, 5, 1])
        )
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        other = self.ids_cls(pd.MultiIndex.from_arrays([["a"], [0]]))
        expected = self.ids_cls(self._circuit_ids(["a", "a", "b", "a", "a"], [0, 1, 0, 2, 0]))
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        test_obj = self.ids_cls(_multi_index())
        other = self.ids_cls(pd.MultiIndex.from_arrays([["c", "b", "c"], [0, 5, 1]]))
        test_obj.append(other, inplace=True)
        expected = self.ids_cls(
            self._circuit_ids(["a", "a", "b", "a", "c", "b", "c"], [0, 1, 0, 2, 0, 5, 1])
        )
        assert test_obj == expected

    def test_intersection(self):
        test_obj = self.ids_cls.from_tuples(_multi_index(), sort_index=False)
        other = self.ids_cls.from_tuples([("b", 0), ("a", 3), ("a", 2)], sort_index=False)
        expected = self.ids_cls.from_tuples([("a", 2), ("b", 0)])
        res = test_obj.intersection(other)

        # res should be sorted when inplace=False
        assert res == expected
        assert test_obj != expected

        res = test_obj.intersection(other, inplace=True)
        assert res is None

        # test_obj index should not be sorted when inplace=True
        assert test_obj != expected
        assert all(expected.index == test_obj.index.sort_values())

    def test_sample(self):
        with patch("numpy.random.choice", return_value=np.array([0, 3])):
            tested = self.test_obj_unsorted.sample(2, inplace=False)
            assert tested == self.ids_cls(self._circuit_ids(["a", "b"], [0, 0]))

        tested = self.test_obj_unsorted.sample(2, inplace=False)
        assert len(tested) == 2

        tested = self.test_obj_unsorted.sample(25, inplace=False)
        assert len(tested) == len(_multi_index())

        values = _multi_index()
        test_obj = self.ids_cls(values)
        assert len(test_obj) == 4
        test_obj.sample(1, inplace=True)
        assert len(test_obj) == 1
        assert len(self.ids_cls(pd.MultiIndex.from_arrays([[], []])).sample(2)) == 0

    def test_limit(self):
        tested = self.test_obj_sorted.limit(2, inplace=False)
        assert len(tested) == 2
        assert tested == self.ids_cls(self._circuit_ids(["a", "a"], [0, 1]))
        assert len(self.ids_cls(pd.MultiIndex.from_arrays([[], []])).limit(2)) == 0

    def test_unique(self):
        tested = self.ids_cls.from_dict({"a": [0, 0, 1], "b": [1, 2, 2]}).unique()
        expected = self.ids_cls.from_dict({"a": [0, 1], "b": [1, 2]})
        assert tested == expected
        assert len(self.ids_cls(pd.MultiIndex.from_arrays([[], []])).unique()) == 0

    def test_tolist(self):
        expected = [("a", 0), ("a", 1), ("b", 0), ("a", 2)]
        assert self.test_obj_unsorted.tolist() == expected

    def test_combined_operators(self):
        tested = self.test_obj_sorted.copy()
        tested.filter_population("a", inplace=True)
        tested.limit(3, inplace=True)
        assert tested == self.ids_cls(self._circuit_ids(["a", "a", "a"], [0, 1, 2]))

        tested = self.test_obj_sorted.copy()
        tested = tested.filter_population("a").limit(3)
        assert tested == self.ids_cls(self._circuit_ids(["a", "a", "a"], [0, 1, 2]))

    def test_equal(self):
        values = _multi_index()
        test_obj = self.ids_cls(values)
        other = self.ids_cls(values)
        assert test_obj == other

        diff_values = pd.MultiIndex.from_arrays([["a", "a", "b"], [0, 1, 0]])
        other = self.ids_cls(diff_values)
        assert test_obj != other

        # different object
        assert test_obj != 1
        # python2 complains
        assert not test_obj.__eq__(1)

        # same object
        test_obj = self.ids_cls(values)
        same_obj = test_obj
        assert test_obj == same_obj

    def test___call__(self):
        # the call is used so we can use the CircuitIds directly in a dataframe
        data = [0, 1, 2, 3]
        df = pd.DataFrame(data={"data": data}, index=_multi_index())
        # test_obj_unsorted = ['a', 'a', 'b', 'a'], [0, 1, 0, 2]
        assert df.loc[self.test_obj_unsorted, "data"].tolist() == data
        # test_obj_unsorted = ['a', 'a', 'a', 'b'], [0, 1, 2, 0] returned value should swap 2 and 3
        assert df.loc[self.test_obj_sorted, "data"].tolist() == [0, 1, 3, 2]

    def test_printing(self):
        tested = self.test_obj_unsorted.__repr__()
        class_name = self.test_obj_sorted.__class__.__name__
        expected = """{}([('a', 0),
            ('a', 1),
            ('b', 0),
            ('a', 2)],
           names=['population', '{}'])""".format(
            class_name, self.id_name
        )

        assert tested == expected
        assert repr(self.test_obj_sorted) == str(self.test_obj_sorted)

    def test_roundtrip(self):
        with setup_tempdir() as tmp_dir:
            output = Path(tmp_dir, "output.csv")
            self.test_obj_sorted.to_csv(str(output))
            new = self.ids_cls.from_csv(str(output))
            assert self.test_obj_sorted == new

    def test___iter__(self):
        assert list(self.test_obj_sorted) == [("a", 0), ("a", 1), ("a", 2), ("b", 0)]
        assert all(type(cid) == self.id_cls for cid in self.test_obj_sorted)

    def test___getitem__(self):
        assert self.test_obj_sorted[0] == self.id_cls("a", 0)


class TestCircuitEdgeIds(TestCircuitNodeIds):
    @property
    def ids_cls(self):
        return test_module.CircuitEdgeIds

    @property
    def id_cls(self):
        return CircuitEdgeId

    @property
    def id_name(self):
        return "edge_ids"
