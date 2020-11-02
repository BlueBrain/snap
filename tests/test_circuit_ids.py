from mock import patch
import pandas as pd
import pandas.testing as pdt
import numpy.testing as npt
import numpy as np

import bluepysnap.circuit_ids as test_module
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from utils import setup_tempdir


def _create_index(populations, ids):
    index = pd.MultiIndex.from_arrays([populations, ids])
    index.names = ['population', 'node_ids']
    return index


def circuit_node_ids():
    return pd.MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 0, 2]])


class TestCircuitNodeIds:
    def setup(self):
        self.test_obj_unsorted = test_module.CircuitNodeIds(circuit_node_ids(), sort_index=False)
        self.test_obj_sorted = test_module.CircuitNodeIds(circuit_node_ids())

    def test_create_global_ids(self):
        tested = test_module.CircuitNodeIds.create_ids('a', 0)
        pdt.assert_index_equal(tested.index, _create_index(['a'], [0]))

        tested = test_module.CircuitNodeIds.create_ids('a', [0, 1])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a'], [0, 1]))

        # duplicate ids if id is single int
        tested = test_module.CircuitNodeIds.create_ids(['a', 'b'], 0)
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [0, 0]))

        tested = test_module.CircuitNodeIds.create_ids(['a', 'b'], [0, 1])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [0, 1]))

        # keep ids ordering
        tested = test_module.CircuitNodeIds.create_ids(['a', 'b'], [1, 0], sort_index=False)
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [1, 0]))

        # keep population ordering
        tested = test_module.CircuitNodeIds.create_ids(['b', 'a'], [0, 1], sort_index=False)
        pdt.assert_index_equal(tested.index, _create_index(['b', 'a'], [0, 1]))

        # keep duplicates
        tested = test_module.CircuitNodeIds.create_ids(['a', 'a'], [0, 0])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a'], [0, 0]))
        assert tested.index.size == 2

    def test_copy(self):
        tested = self.test_obj_sorted.copy()
        assert self.test_obj_sorted is not tested
        assert self.test_obj_sorted.index is not tested.index
        assert self.test_obj_sorted == tested

    def test_len(self):
        assert len(self.test_obj_sorted) == len(circuit_node_ids())

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
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj_sorted.filter_population('b')
        pdt.assert_index_equal(tested.index, _create_index(['b'], [0]))

        tested = self.test_obj_sorted.copy()
        tested.filter_population('b', inplace=True)
        pdt.assert_index_equal(tested.index, _create_index(['b'], [0]))

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

    def test_append(self):
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                            [0, 1, 0, 2, 0, 5, 1]))
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['a'], [0]]))
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'a'],
                                                            [0, 1, 0, 2, 0]))
        assert self.test_obj_sorted.append(other, inplace=False) == expected

        test_obj = test_module.CircuitNodeIds(circuit_node_ids())
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        test_obj.append(other, inplace=True)
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                            [0, 1, 0, 2, 0, 5, 1]))
        assert test_obj == expected

    def test_sample(self):
        with patch("numpy.random.choice", return_value=np.array([0, 3])):
            tested = self.test_obj_unsorted.sample(2, inplace=False)
            assert tested == test_module.CircuitNodeIds(_create_index(['a', 'b'], [0, 0]))

        tested = self.test_obj_unsorted.sample(2, inplace=False)
        assert len(tested) == 2

        tested = self.test_obj_unsorted.sample(25, inplace=False)
        assert len(tested) == len(circuit_node_ids())

        values = circuit_node_ids()
        test_obj = test_module.CircuitNodeIds(values)
        assert len(test_obj) == 4
        test_obj.sample(1, inplace=True)
        assert len(test_obj) == 1

    def test_limit(self):
        tested = self.test_obj_sorted.limit(2, inplace=False)
        assert len(tested) == 2
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a'], [0, 1]))

    def test_combined_operators(self):
        tested = self.test_obj_sorted.copy()
        tested.filter_population("a", inplace=True)
        tested.limit(3, inplace=True)
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj_sorted.copy()
        tested = tested.filter_population("a").limit(3)
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a', 'a'], [0, 1, 2]))

    def test_equal(self):
        values = circuit_node_ids()
        test_obj = test_module.CircuitNodeIds(values)
        other = test_module.CircuitNodeIds(values)
        assert test_obj == other

        diff_values = pd.MultiIndex.from_arrays([['a', 'a', 'b'], [0, 1, 0]])
        other = test_module.CircuitNodeIds(diff_values)
        assert test_obj != other

        # different object
        assert test_obj != 1

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
