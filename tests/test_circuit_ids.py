import pytest
from mock import patch
import pandas as pd
import pandas.testing as pdt
import numpy.testing as npt
import numpy as np

from pathlib import Path


from bluepysnap.exceptions import BluepySnapError
import bluepysnap.circuit_ids as test_module

from utils import setup_tempdir


def _circuit_ids(populations, ids):
    index = pd.MultiIndex.from_arrays([populations, ids])
    index.names = ['population', 'node_ids']
    return index


def _multi_index():
    return pd.MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 0, 2]])


class TestCircuitId:
    def setup(self):
        self.test_obj = test_module.CircuitNodeId("pop", 1)

    def test_init(self):
        assert isinstance(self.test_obj, test_module.CircuitNodeId)
        assert isinstance(self.test_obj, tuple)

    def test_accessors(self):
        assert self.test_obj.population == "pop"
        assert self.test_obj.id == 1


class TestCircuitNodeIds:
    def setup(self):
        self.test_obj_unsorted = test_module.CircuitNodeIds(_multi_index(), sort_index=False)
        self.test_obj_sorted = test_module.CircuitNodeIds(_multi_index())

    def test_init(self):
        values = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 0, 2]])
        tested = test_module.CircuitNodeIds(values, sort_index=False)
        assert tested.index.names == ["population", "node_ids"]
        npt.assert_equal(tested.index.values, values.values)

        tested = test_module.CircuitNodeIds(values, sort_index=True)
        assert tested.index.names == ["population", "node_ids"]
        npt.assert_equal(tested.index.values, np.sort(values.values))

        with pytest.raises(BluepySnapError):
            test_module.CircuitNodeIds(1)

    def test_from_arrays(self):
        tested = test_module.CircuitNodeIds.from_arrays(['a', 'b'], [0, 1])
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [0, 1]))

        # keep ids ordering
        tested = test_module.CircuitNodeIds.from_arrays(['a', 'b'], [1, 0], sort_index=False)
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [1, 0]))

        # keep population ordering
        tested = test_module.CircuitNodeIds.from_arrays(['b', 'a'], [0, 1], sort_index=False)
        pdt.assert_index_equal(tested.index, _circuit_ids(['b', 'a'], [0, 1]))

        # keep duplicates
        tested = test_module.CircuitNodeIds.from_arrays(['a', 'a'], [0, 0])
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'a'], [0, 0]))
        assert tested.index.size == 2

        with pytest.raises(BluepySnapError):
            test_module.CircuitNodeIds.from_arrays(['a', 'a', 'a'], [0, 0])

        with pytest.raises(BluepySnapError):
            test_module.CircuitNodeIds.from_arrays(['a', 'a'], [0, 0, 0])

    def test_create_ids(self):
        tested = test_module.CircuitNodeIds.from_dict({'a': [0]})
        pdt.assert_index_equal(tested.index, _circuit_ids(['a'], [0]))

        tested = test_module.CircuitNodeIds.from_dict({'a': [0, 1]})
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'a'], [0, 1]))

        tested = test_module.CircuitNodeIds.from_dict({'a': [0], 'b': [0]})
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [0, 0]))

        tested = test_module.CircuitNodeIds.from_dict({'a': [0], 'b': [1]})
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [0, 1]))

        # keep duplicates
        tested = test_module.CircuitNodeIds.from_dict({'a': [0, 0]})
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'a'], [0, 0]))
        assert tested.index.size == 2

    def test_from_tuples(self):
        tested = test_module.CircuitNodeIds.from_tuples([test_module.CircuitNodeId("a", 0),
                                                         test_module.CircuitNodeId("b", 1)])
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [0, 1]))
        tested = test_module.CircuitNodeIds.from_tuples([("a", 0), ("b", 1)])
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'b'], [0, 1]))

    def test_copy(self):
        tested = self.test_obj_sorted.copy()
        assert self.test_obj_sorted is not tested
        assert self.test_obj_sorted.index is not tested.index
        assert self.test_obj_sorted == tested

    def test_len(self):
        assert len(self.test_obj_sorted) == len(_multi_index())

    def test__locate(self):
        tested = self.test_obj_sorted._locate('a')
        npt.assert_equal(tested, [0, 1, 2])
        tested = self.test_obj_unsorted._locate('a')
        npt.assert_equal(tested, [0, 1, 3])

        tested = self.test_obj_sorted._locate('b')
        npt.assert_equal(tested, [3])
        tested = self.test_obj_unsorted._locate('b')
        npt.assert_equal(tested, [2])

    def test_filter_population(self):
        tested = self.test_obj_sorted.filter_population('a')
        pdt.assert_index_equal(tested.index, _circuit_ids(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj_sorted.filter_population('b')
        pdt.assert_index_equal(tested.index, _circuit_ids(['b'], [0]))

        tested = self.test_obj_sorted.copy()
        tested.filter_population('b', inplace=True)
        pdt.assert_index_equal(tested.index, _circuit_ids(['b'], [0]))

    def test_get_populations(self):
        tested = self.test_obj_sorted.get_populations()
        npt.assert_equal(tested, ['a', 'a', 'a', 'b'])
        tested = self.test_obj_unsorted.get_populations()
        npt.assert_equal(tested, ['a', 'a', 'b', 'a'])

        tested = self.test_obj_sorted.get_populations(unique=True)
        npt.assert_equal(tested, ['a', 'b'])
        tested = self.test_obj_unsorted.get_populations(unique=True)
        npt.assert_equal(tested, ['a', 'b'])

    def test_get_ids(self):
        tested = self.test_obj_sorted.get_ids()
        npt.assert_equal(tested, [0, 1, 2, 0])
        tested = self.test_obj_unsorted.get_ids()
        npt.assert_equal(tested, [0, 1, 0, 2])

        tested = self.test_obj_sorted.get_ids(unique=True)
        npt.assert_equal(tested, [0, 1, 2])
        tested = self.test_obj_unsorted.get_ids(unique=True)
        npt.assert_equal(tested, [0, 1, 2])

    def test_sort(self):
        obj = test_module.CircuitNodeIds(_multi_index(), sort_index=False)
        expected = pd.MultiIndex.from_arrays([['a', 'a', 'a', 'b'], [0, 1, 2, 0]])
        npt.assert_equal(obj.sort().index.values, expected.values)
        obj.sort(inplace=True)
        npt.assert_equal(obj.index.values, expected.values)

    def test_append(self):
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        expected = test_module.CircuitNodeIds(_circuit_ids(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                           [0, 1, 0, 2, 0, 5, 1]))
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['a'], [0]]))
        expected = test_module.CircuitNodeIds(_circuit_ids(['a', 'a', 'b', 'a', 'a'],
                                                           [0, 1, 0, 2, 0]))
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        test_obj = test_module.CircuitNodeIds(_multi_index())
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        test_obj.append(other, inplace=True)
        expected = test_module.CircuitNodeIds(_circuit_ids(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                           [0, 1, 0, 2, 0, 5, 1]))
        assert test_obj == expected

    def test_sample(self):
        with patch("numpy.random.choice", return_value=np.array([0, 3])):
            tested = self.test_obj_unsorted.sample(2, inplace=False)
            assert tested == test_module.CircuitNodeIds(_circuit_ids(['a', 'b'], [0, 0]))

        tested = self.test_obj_unsorted.sample(2, inplace=False)
        assert len(tested) == 2

        tested = self.test_obj_unsorted.sample(25, inplace=False)
        assert len(tested) == len(_multi_index())

        values = _multi_index()
        test_obj = test_module.CircuitNodeIds(values)
        assert len(test_obj) == 4
        test_obj.sample(1, inplace=True)
        assert len(test_obj) == 1

    def test_limit(self):
        tested = self.test_obj_sorted.limit(2, inplace=False)
        assert len(tested) == 2
        assert tested == test_module.CircuitNodeIds(_circuit_ids(['a', 'a'], [0, 1]))

    def test_tolist(self):
        expected = [('a', 0), ('a', 1), ('b', 0), ('a', 2)]
        assert self.test_obj_unsorted.tolist() == expected

    def test_combined_operators(self):
        tested = self.test_obj_sorted.copy()
        tested.filter_population("a", inplace=True)
        tested.limit(3, inplace=True)
        assert tested == test_module.CircuitNodeIds(_circuit_ids(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj_sorted.copy()
        tested = tested.filter_population("a").limit(3)
        assert tested == test_module.CircuitNodeIds(_circuit_ids(['a', 'a', 'a'], [0, 1, 2]))

    def test_equal(self):
        values = _multi_index()
        test_obj = test_module.CircuitNodeIds(values)
        other = test_module.CircuitNodeIds(values)
        assert test_obj == other

        diff_values = pd.MultiIndex.from_arrays([['a', 'a', 'b'], [0, 1, 0]])
        other = test_module.CircuitNodeIds(diff_values)
        assert test_obj != other

        # different object
        assert test_obj != 1
        # python2 complains
        assert not test_obj.__eq__(1)

        # same object
        test_obj = test_module.CircuitNodeIds(values)
        same_obj = test_obj
        assert test_obj == same_obj

    def test___iter__(self):
        assert list(self.test_obj_sorted) == [('a', 0), ('a', 1), ('a', 2), ('b', 0)]
        assert all(type(cid) == test_module.CircuitNodeId for cid in self.test_obj_sorted)

    def test___call__(self):
        # the call is used so we can use the CircuitNodeIds directly in a dataframe
        data = [0, 1, 2, 3]
        df = pd.DataFrame(data={"data": data}, index=_multi_index())
        # test_obj_unsorted = ['a', 'a', 'b', 'a'], [0, 1, 0, 2]
        assert df.loc[self.test_obj_unsorted, "data"].tolist() == data
        # test_obj_unsorted = ['a', 'a', 'a', 'b'], [0, 1, 2, 0] returned value should swap 2 and 3
        assert df.loc[self.test_obj_sorted, "data"].tolist() == [0, 1, 3, 2]

    def test_printing(self):
        tested = self.test_obj_unsorted.__repr__()
        expected = """CircuitNodeIds([('a', 0),
            ('a', 1),
            ('b', 0),
            ('a', 2)],
           names=['population', 'node_ids'])"""

        assert tested == expected
        assert self.test_obj_sorted.__repr__() == self.test_obj_sorted.__str__()

    def test_roundtrip(self):
        with setup_tempdir() as tmp_dir:
            output = Path(tmp_dir, "output.csv")
            self.test_obj_sorted.to_csv(str(output))
            new = test_module.CircuitNodeIds.from_csv(str(output))
            assert self.test_obj_sorted == new
